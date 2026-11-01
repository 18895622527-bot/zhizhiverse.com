'use strict';
/* ============================================================
 * zzCrop —— 照片自动预处理：摆正 / 裁剪 / 去花边 / 重新构图 / 换背景
 *
 * 【链路】
 *   小程序给一个 fileID
 *   → 这里换成临时地址，交给腾讯云 SCF 上的 zz-crop（Python，OpenCV + MediaPipe）
 *   → 它回一张只剩照片本身的图
 *   → 存回云存储，回新的 fileID
 *
 * 【为什么传地址不传字节】
 *   SCF 同步调用的请求体上限 6MB，手机原图 base64 之后很容易超。
 *   照片本来就在腾讯云存储上，给个临时地址让 SCF 自己去下，同一朵云里几十毫秒。
 *   ⚠️ 所以这个函数【不】限制图片大小 —— 和 zzColorBaidu 的 3MB 上限不一样。
 *
 * 【fix 为空不是失败】
 *   SCF 没认出照片的边时回 fix: ''。这时回原来的 fileID，让小程序走用户自己框的老路。
 *   ⚠️ 调用方要看 fix 来决定要不要跳过框选页，不能只看 code。
 *
 * 【config.json】{ "endpoint": "<SCF 的 https 地址>", "token": "<跟 SCF 环境变量 ZZ_TOKEN 一样>" }
 *   不要写进这个文件，不要提交到任何地方。
 * ============================================================ */

let cfg = {};
try { cfg = require('./config.json') || {}; } catch (e) { cfg = {}; }

function err(msg, detail) {
	return { code: 1, msg: msg, detail: String(detail || '').slice(0, 400) };
}

/* fileID → 临时 https 地址。已经是 https 就原样用 */
async function tempUrl(fileID) {
	let url = String(fileID || '').trim();
	if (!url) throw new Error('没给图片');
	if (url.indexOf('cloud://') === 0) {
		const t = await uniCloud.getTempFileURL({ fileList: [url] });
		const f = ((t && t.fileList) || [])[0];
		url = String((f && f.tempFileURL) || '');
		if (!url) throw new Error('云存储换不出临时地址');
	}
	if (url.indexOf('http') !== 0) throw new Error('图片地址不认识：' + url.slice(0, 80));
	return url;
}

async function callScf(url, opt) {
	if (!cfg.endpoint || !cfg.token) throw new Error('config.json 里没填 endpoint / token');
	const r = await uniCloud.httpclient.request(cfg.endpoint, {
		method: 'POST',
		/* ⚠️ contentType: 'json' 让 urllib 自己把 data 序列化成 JSON 并带上 Content-Type。
		 *    和 zzColorBaidu 那边的 form 表单不一样，别照抄。 */
		contentType: 'json',
		dataType: 'json',
		headers: { 'X-ZZ-Token': cfg.token },
		data: { url: url, beautify: opt.beautify !== false, border: opt.border !== false },
		/* 冷启动要加载 250MB 依赖，第一次可能 10 秒；热的时候 1 秒内 */
		timeout: 60000
	});
	const d = (r && r.data) || {};
	if (r && r.status === 403) throw new Error('SCF 说没有权限：两边 token 不一样');
	if (!d.ok) throw new Error(d.msg || ('SCF 回了 ' + (r && r.status)));
	return d;
}

async function saveToCloud(buf) {
	const name = 'mp-liuying/crop_' + Date.now() + '_' + Math.floor(Math.random() * 1e6) + '.jpg';
	const r = await uniCloud.uploadFile({ cloudPath: name, fileContent: buf });
	const fid = (r && r.fileID) || '';
	if (!fid) throw new Error('结果图没存进云存储');
	return fid;
}

exports.main = async function (event) {
	const ev = event || {};
	const op = String(ev.op || (ev.fileID ? 'crop' : 'ping'));

	if (op === 'ping') {
		if (!cfg.endpoint || !cfg.token) return err('config.json 里没填 endpoint / token');
		try {
			/* 故意不给图，期望 SCF 回 400「没给图片」—— 说明地址、token、依赖都通了 */
			const r = await uniCloud.httpclient.request(cfg.endpoint, {
				method: 'POST', contentType: 'json', dataType: 'json',
				headers: { 'X-ZZ-Token': cfg.token }, data: {}, timeout: 30000
			});
			const d = (r && r.data) || {};
			if (r.status === 403) return err('token 不一样', JSON.stringify(d));
			if (r.status === 400 && d.msg === '没给图片') return { code: 0, msg: '通了', endpointTail: cfg.endpoint.slice(-24) };
			return err('SCF 回了 ' + r.status, JSON.stringify(d).slice(0, 300));
		} catch (e) {
			return err('SCF 连不上', (e && e.message) || e);
		}
	}

	if (op !== 'crop') return err('op 只支持 ping / crop');

	let url;
	try { url = await tempUrl(ev.fileID); }
	catch (e) { return err('这张照片我读不出来', (e && e.message) || e); }

	let d;
	try { d = await callScf(url, ev); }
	catch (e) { return err('自动裁剪这会儿不行，让用户自己框', (e && e.message) || e); }

	/* 没认出边：不是失败，回原图，fix 空串 */
	if (!d.fix) return { code: 0, fileID: ev.fileID, fix: '', w: d.w, h: d.h, ms: d.ms };

	try {
		const fid = await saveToCloud(Buffer.from(d.image, 'base64'));
		return { code: 0, fileID: fid, fix: d.fix, w: d.w, h: d.h, ratio: d.ratio, ms: d.ms };
	} catch (e) {
		return err('裁好了但没存住', (e && e.message) || e);
	}
};
