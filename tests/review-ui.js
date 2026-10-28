// Developer-only browser checks. Appended to a generated, self-contained report.
// Runs in existing Chromium; no browser/package dependency enters Tide runtime.
(async()=>{
  const results=[];
  function check(value,label){if(!value)throw Error(label);results.push(label)}
  function click(id){document.getElementById(id).click()}
  try {
    check(document.getElementById('hints').children.length===0,'hints hidden initially');
    click('reveal');check(document.getElementById('hints').children.length>0,'observations revealed');
    document.querySelector('#hints button').click();
    check(!document.getElementById('rating').hidden,'finding selected');
    const assessment=document.getElementById('assessment');assessment.value='intentional';assessment.dispatchEvent(new Event('change'));
    const reason=document.getElementById('reason');reason.value='<img src=x onerror=alert(1)> intentional action';reason.dispatchEvent(new Event('input'));
    document.getElementById('newNote').value='Tool missed the intended recovery';click('addNote');
    check(document.getElementById('notes').textContent.includes('Tool missed'),'timestamped free feedback');
    check(document.getElementById('notes').querySelectorAll('img').length===0,'notes rendered as text');
    let exported;URL.createObjectURL=blob=>{exported=blob;return 'blob:test'};URL.revokeObjectURL=()=>{};HTMLAnchorElement.prototype.click=()=>{};
    click('export');const f=JSON.parse(await exported.text());const key=Object.keys(f.cases)[0];
    check(f.cases[key].notes.length===1,'notes exported');check(Object.values(f.cases[key].observations)[0].assessment==='intentional','intentional rating exported');
    check(f.cases[key].notes[0].nearbyObservationIds.length>0,'feedback linked to nearby observations');
    async function load(v){const transfer=new DataTransfer();transfer.items.add(new File([JSON.stringify(v)],'feedback.json',{type:'application/json'}));document.getElementById('import').files=transfer.files;await document.getElementById('import').onchange()}
    await load(f);check(document.getElementById('status').textContent.startsWith('Feedback imported'),'feedback roundtrip');
    const changed=structuredClone(f);changed.identity=[];await load(changed);check(document.getElementById('status').textContent.startsWith('Import refused'),'mismatched source identity refused');
    const invalid=structuredClone(f);invalid.cases[key].notes[0].frame=999999;await load(invalid);check(document.getElementById('status').textContent.startsWith('Import refused'),'invalid feedback timestamps refused');
    click('reveal');check(document.getElementById('hints').children.length===0,'hints can be hidden again');
    const slider=document.getElementById('scrub');slider.value=3;slider.dispatchEvent(new Event('input'));check(document.getElementById('time').textContent.includes('frame 3/'),'frame scrubbing');
    click('forward');check(document.getElementById('time').textContent.includes('frame 4/'),'frame stepping');
    document.body.setAttribute('data-review-test','pass');
  }catch(e){document.body.setAttribute('data-review-test','FAIL: '+e.message)}
  const output=document.createElement('pre');output.id='test-results';output.textContent=JSON.stringify(results);document.body.append(output);
})();
