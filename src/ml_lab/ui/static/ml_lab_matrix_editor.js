/** ML_Lab IO/validation adapter for the one PAH_Shared matrix editor. */
(() => {
  const root = document.getElementById('matrix-draft');
  if (!root) return;
  const dtypes = JSON.parse(root.dataset.dtypes || '[]');
  const sourceHash = root.dataset.sourceHash;
  const fetchJson = async (url, options) => {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) throw Error(payload.error || `Request failed (${response.status})`);
    return payload;
  };
  import(root.dataset.sharedEditorUrl).then(({mountMatrixEditor}) => {
    mountMatrixEditor(root, {
      loadPage: ({page,pageSize}) => {
        const params = new URLSearchParams({page:String(page),page_size:String(pageSize)});
        return fetchJson(`${root.dataset.rowsUrl}?${params}`,{headers:{Accept:'application/json'}});
      },
      saveDraft: ({edits}) => fetchJson(root.dataset.saveUrl, {
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({source_sha256:sourceHash,edits}),
      }),
      validateCell: (value,col) => {
        if (value === '') return null;
        const type = dtypes[col] || '';
        if (/^(u?int)/.test(type) && !/^[+-]?\d+$/.test(value)) return 'This column requires an integer. Escape cancels the edit.';
        if (type.startsWith('float') && !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$|^[+-]?(?:nan|inf|infinity)$/i.test(value)) return 'This column requires a numeric value. Escape cancels the edit.';
        return null;
      },
      onSaved: (payload, box) => {
        const a=(url,label)=>{const link=document.createElement('a');link.href=url;link.textContent=label;box.append(document.createTextNode(' · '),link);};
        if(payload.source_url)a(payload.source_url,'Open derived dataset');
        if(payload.provenance_url)a(payload.provenance_url,'View provenance');
        if(payload.experiment_url)a(payload.experiment_url,'Use in experiment');
      },
    });
  }).catch(error => {root.textContent=`Shared matrix editor unavailable: ${error.message}`;});
})();
