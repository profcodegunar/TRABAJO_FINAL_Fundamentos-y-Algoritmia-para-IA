
const sidebar=document.getElementById('sidebar');
document.getElementById('menu')?.addEventListener('click',()=>sidebar.classList.toggle('open'));
document.querySelectorAll('.nav-item').forEach(a=>a.addEventListener('click',()=>sidebar.classList.remove('open')));
document.querySelectorAll('.copy').forEach(btn=>btn.addEventListener('click',async()=>{
 const el=document.getElementById(btn.dataset.target); const text=el.innerText;
 try{await navigator.clipboard.writeText(text); const old=btn.textContent; btn.textContent='Copiado'; setTimeout(()=>btn.textContent=old,1400)}
 catch(e){btn.textContent='Seleccione y copie'; setTimeout(()=>btn.textContent='Copiar código',1800)}
}));
const sections=[...document.querySelectorAll('.searchable')];
const nav=[...document.querySelectorAll('.nav-item')];
function update(){const max=document.documentElement.scrollHeight-innerHeight; const p=max?Math.round(scrollY/max*100):0; document.getElementById('bar').style.width=p+'%';document.getElementById('pct').textContent=p+'%'; let current='';sections.forEach(s=>{if(s.getBoundingClientRect().top<=120)current=s.id});nav.forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+current));}
addEventListener('scroll',update,{passive:true});update();
const search=document.getElementById('search');search.addEventListener('input',()=>{const q=search.value.trim().toLowerCase();sections.forEach(s=>s.classList.toggle('hidden-by-search',q && !s.innerText.toLowerCase().includes(q)));});
