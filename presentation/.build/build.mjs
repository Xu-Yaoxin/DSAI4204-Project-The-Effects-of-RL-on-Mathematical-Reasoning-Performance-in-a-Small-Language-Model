import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
import {GlobalFonts} from '@napi-rs/canvas';
import {applyPresentationChartFont,finalizePresentation} from '/Users/xu./.codex/plugins/cache/openai-primary-runtime/presentations/26.1030.10813/skills/presentations/container_tools/artifact_tool_utils.mjs';

const root=path.resolve('../');
const skill='/Users/xu./.codex/plugins/cache/openai-primary-runtime/presentations/26.1030.10813/skills/presentations';
const fonts=GlobalFonts.families.map(x=>x.family);
const font=fonts.includes('Arial')?'Arial':fonts.includes('Aptos')?'Aptos':fonts.includes('Helvetica')?'Helvetica':fonts[0];
console.log('FONT',font);
const summary=JSON.parse(await fs.readFile(path.join(root,'../results/final/summary.json'),'utf8'));
const notes=await fs.readFile(path.join(root,'SPEAKER_NOTES.md'),'utf8');
const noteSections=notes.split(/\n## \d+\. /).slice(1);
const P=Presentation.create({slideSize:{width:1280,height:720}});
const C={bg:'#F6F3EC',ink:'#132D3A',muted:'#53636D',base:'#8495A8',rl:'#007E7B',pale:'#DCECE7',line:'#D5DDD9',orange:'#BD582B',white:'#FFFFFF'};
const titles=['Does RL Make a 2B\nModel More Reliable?','Higher reliability, wider observed coverage','Reliability and observed coverage','Independent data, paired evaluation','Training with verifiable rewards','Development results select step 384','What pass@1, pass@4 and pass@16 measure','RL adds 575 correct responses','Single-answer reliability gains the most','RL covers 17 more test questions','One scorer, the same original answers','Better results under a limited protocol','References and experiment sources'];
const subs=['A controlled study on MATH Level 4','Final test on 200 selected MATH Level 4 questions','A small-scale adaptation of Yue et al. (2025)','Same 200 test questions and 16 samples per question for each model','LoRA with group relative policy optimization (GRPO)','128 development questions × 4 samples = 512 responses per checkpoint','At least one confirmed-correct answer within the sampling budget','6,400 answers across 200 paired test questions','RL minus Base (pp), with 95% paired confidence intervals','Observed coverage within 16 attempts per question','Post-hoc v4 scoring, with no new generations','Stops, scoring uncertainty and unequal generation speeds','Reference appendix, outside the 10-minute talk'];

for(let i=0;i<13;i++){
 const s=P.slides.add();
 const dark=i===0||i===2;
 s.background.fill=dark?C.ink:C.bg;
 const ink=dark?C.bg:C.ink;
 const muted=dark?'#BAD0D5':C.muted;
 const items=[];
 const shapes=[];
 const charts=[];
 if(i===0){
  items.push(['DSAI4207 GROUP PROJECT',68,56,900,36,22,'#87CFC5',true],
   [titles[i],68,185,1130,190,70,C.white,true],
   [subs[i],72,413,1050,56,30,'#C4D8DA',false],
   ['Qwen3.5-2B-Base  /  LoRA + GRPO',72,548,1050,40,24,'#87CFC5',false],
   ['10-minute research presentation',72,631,1000,30,19,'#BAD0D5',false]);
 }else{
  items.push([titles[i],64,43,1154,68,42,ink,true],
   [subs[i],66,119,1148,54,23,muted,false]);
 }
 if(i===1){
  charts.push(['bar',{position:{left:64,top:225,width:780,height:360},categories:['pass@1','pass@16'],series:[{name:'Base',values:[0.4353125,0.855],fill:C.base},{name:'RL',values:[0.615,0.94],fill:C.rl}],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:21}},barOptions:{direction:'column',grouping:'clustered',gapWidth:80},xAxis:{textStyle:{fontSize:23},majorGridlines:null},yAxis:{min:0,max:1,majorUnit:0.2,numberFormatCode:'0%',textStyle:{fontSize:19},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00%',textStyle:{fontSize:22,bold:true,fill:C.ink}}}]);
  items.push(['+17.97',900,231,310,83,62,C.rl,true],['percentage points\nin pass@1',904,318,290,70,25,C.ink,false],['+8.50',900,435,310,83,62,C.rl,true],['percentage points\nin pass@16',904,519,290,70,25,C.ink,false]);
 }
 if(i===2){
  items.push(['ONE ATTEMPT',70,239,530,38,24,'#87CFC5',true],['How often does one\nanswer succeed?',70,299,530,125,40,C.white,true],['Reliability',70,470,530,45,28,'#BAD0D5',false],
   ['16 ATTEMPTS',700,239,510,38,24,'#87CFC5',true],['How many questions\nare covered?',700,299,510,125,40,C.white,true],['Finite-budget coverage',700,470,510,45,28,'#BAD0D5',false],
   ['Our model, training budget and sampling range differ from the source paper.',70,576,1130,60,24,'#BAD0D5',false]);
  shapes.push(['rect',636,238,2,278,'#38535D']);
 }
 if(i===3){
  const ds=[['512','TRAIN','Update LoRA parameters',76,C.rl],['128','DEVELOPMENT','Choose the checkpoint',484,C.ink],['200','TEST','Measure Base versus RL',894,C.ink]];
  for(const [n,label,desc,x,color] of ds){items.push([n,x,220,320,100,76,color,true],[label,x,330,320,36,23,color,true],[desc,x,379,318,70,24,C.muted,false]);}
  shapes.push(['rect',70,476,1138,2,C.line]);
  items.push(['200 × 16 × 2 = 6,400 answers',75,515,1140,65,44,C.ink,true],['Qwen3.5-2B-Base  ·  temperature 0.6  ·  top-p 0.95',75,594,1100,35,24,C.muted,false]);
 }
 if(i===4){
  const flow=[['Question',65,207],['8 sampled\nanswers',310,207],['Verifier\n0 or 1',555,207],['Relative\nrewards',800,207],['LoRA\nupdate',1045,207]];
  for(let j=0;j<flow.length;j++){
   const [t,x,y]=flow[j]; shapes.push(['rect',x,y,176,104,j===2?C.rl:C.pale]);items.push([t,x+9,y+21,158,75,24,j===2?C.white:C.ink,true]);
   if(j<flow.length-1)shapes.push(['rightArrow',x+186,y+39,47,24,C.base]);
  }
  items.push(['10.9M',70,383,370,85,62,C.rl,true],['trainable parameters',75,470,370,42,25,C.ink,false],['3 epochs / 384 updates',528,385,650,56,35,C.ink,true],['Rank 16   ·   Learning rate 0.000005\nOne training seed   ·   KL coefficient 0\nTraining answer limit: 2,048 tokens',531,457,650,132,26,C.muted,false],['Binary answer reward; no SFT or teacher-generated reasoning.',75,601,1135,42,25,C.ink,false]);
 }
 if(i===5){
  charts.push(['line',{position:{left:64,top:221,width:790,height:357},categories:['Step 128','Step 256','Step 384'],series:[{name:'Development response success',values:[255/512,299/512,309/512],line:{fill:C.rl,width:4},marker:{symbol:'circle',size:11,fill:C.rl}}],hasLegend:false,xAxis:{textStyle:{fontSize:22},majorGridlines:null},yAxis:{min:0,max:0.7,majorUnit:0.1,numberFormatCode:'0%',textStyle:{fontSize:18},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:true,position:'top',numberFormatCode:'0.00%',textStyle:{fontSize:24,bold:true,fill:C.rl}}}]);
  items.push(['384',930,260,280,96,78,C.rl,true],['selected step',934,363,270,40,28,C.ink,false],['309 / 512 correct',918,441,295,50,27,C.ink,true],['Development scorer v3. Test outcomes did not select the model.',75,607,1130,45,25,C.muted,false]);
 }
 if(i===6){
  for(let j=0;j<16;j++)shapes.push(['ellipse',78+j*69,232,43,43,j<8?C.rl:'#D8DFDF']);
  items.push(['Real Base example: 8 of 16 answers are correct',78,311,1130,48,29,C.ink,true]);
  const cards=[['pass@1','50%','One sampled answer',80],['pass@4','96.15%','At least one correct in four',497],['pass@16','100%','At least one correct in sixteen',914]];
  for(const [k,v,d,x] of cards)items.push([k,x,398,330,40,25,C.ink,true],[v,x,454,330,85,58,C.rl,true],[d,x,551,318,72,23,C.muted,false]);
 }
 if(i===7){
  charts.push(['bar',{position:{left:58,top:201,width:1157,height:373},categories:['pass@1','pass@4','pass@16'],series:[{name:'Base',values:[summary.metrics.base_pass_at_1.estimate,summary.metrics.base_pass_at_4.estimate,summary.metrics.base_pass_at_16.estimate],fill:C.base},{name:'RL',values:[summary.metrics.rl_pass_at_1.estimate,summary.metrics.rl_pass_at_4.estimate,summary.metrics.rl_pass_at_16.estimate],fill:C.rl}],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:21}},barOptions:{direction:'column',grouping:'clustered',gapWidth:75},xAxis:{textStyle:{fontSize:23},majorGridlines:null},yAxis:{min:0,max:1,majorUnit:0.2,numberFormatCode:'0%',textStyle:{fontSize:18},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00%',textStyle:{fontSize:23,bold:true,fill:C.ink}}}]);
  items.push(['1,393 / 3,200 Base correct',77,606,510,42,29,C.base,true],['1,968 / 3,200 RL correct',705,606,510,42,29,C.rl,true]);
 }
 if(i===8){
  const series=[];
  for(const [j,k] of [1,4,16].entries()){
   const m=summary.metrics['delta_pass_at_'+k],y=3-j;
   series.push({name:'pass@'+k+' 95% CI',xValues:m.ci95.map(v=>100*v),values:[y,y],line:{fill:C.rl,width:5},marker:{symbol:'none'}});
   series.push({name:'pass@'+k+' estimate',xValues:[100*m.estimate],values:[y],line:{fill:'none',width:0},marker:{symbol:'circle',size:14,fill:C.rl}});
   items.push(['pass@'+k,70,229+j*117,180,44,29,C.ink,true],['+'+(100*m.estimate).toFixed(2)+' pp',941,217+j*117,280,43,32,C.rl,true],['['+m.ci95.map(v=>(100*v).toFixed(2)).join(', ')+']',944,260+j*117,275,36,23,C.muted,false]);
  }
  charts.push(['scatter',{position:{left:242,top:182,width:678,height:408},series,hasLegend:false,scatterOptions:{style:'lineWithMarkers'},xAxis:{min:0,max:25,majorUnit:5,numberFormatCode:'0',textStyle:{fontSize:20},majorGridlines:{fill:C.line,width:1}},yAxis:{visible:false,min:0.5,max:3.5,majorGridlines:null},dataLabels:{showValue:false}}]);
  items.push(['2,000 paired bootstrap resamples at question level',72,602,1120,42,25,C.muted,false]);
 }
 if(i===9){
  const cols=[...Array(167).fill(C.rl),...Array(21).fill('#60BDB0'),...Array(4).fill(C.orange),...Array(8).fill('#D3DCDB')];
  for(let j=0;j<200;j++)shapes.push(['rect',73+(j%20)*25,232+Math.floor(j/20)*25,20,20,cols[j]]);
  const lg=[['167','Both solved',C.rl],['21','RL only','#60BDB0'],['4','Base only',C.orange],['8','Neither','#A6B4B4']];
  for(let j=0;j<4;j++){const [n,t,c]=lg[j];items.push([n,694,211+j*94,110,65,45,c,true],[t,832,222+j*94,350,45,29,C.ink,false]);}
  items.push(['Each square represents one test question.',74,510,550,60,23,C.muted,false],['Base 171 / 200     RL 188 / 200',73,602,1120,49,35,C.ink,true]);
 }
 if(i===10){
  items.push(['6,400',72,225,520,100,78,C.rl,true],['original answers preserved',76,333,555,52,29,C.ink,true],['No retraining or new generations\nNo checkpoint reselection\nEvery attempt stays in the denominator',76,421,558,156,27,C.muted,false],
   ['+18',798,227,380,95,76,C.rl,true],['RL answers recovered\nby extraction fixes',801,333,380,90,29,C.ink,true],['Base correct count unchanged\nStopped answers receive no credit\n300-second deadline + repetition guard',707,452,485,136,25,C.muted,false]);
  shapes.push(['rect',650,221,2,353,C.line]);
 }
 if(i===11){
  charts.push(['bar',{position:{left:56,top:218,width:610,height:320},categories:['Timeout','Repetition'],series:[{name:'Base',values:[219,211],fill:C.base},{name:'RL',values:[320,138],fill:C.rl}],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:20}},barOptions:{direction:'column',grouping:'clustered',gapWidth:70},xAxis:{textStyle:{fontSize:22},majorGridlines:null},yAxis:{min:0,max:400,majorUnit:100,numberFormatCode:'0',textStyle:{fontSize:18},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0',textStyle:{fontSize:25,bold:true,fill:C.ink}}}]);
  items.push(['Counts out of 3,200 answers per model',72,554,580,42,22,C.muted,false],['Higher success, especially\non a single attempt',715,223,485,95,34,C.rl,true],['One training seed; selected subset\nSame 300s, unequal token speeds\npass@16 gain: −2.5 to +14 pp',717,349,485,147,25,C.ink,false],['Sensitivity range, not a confidence interval',717,497,485,45,20,C.orange,false],['NEXT  Multiple seeds, matched token budget and independent scoring audit',73,612,1135,46,24,C.ink,true]);
 }
 if(i===12){
  items.push(['RESEARCH PAPER',70,210,1125,35,22,C.rl,true],['Yue, Yang, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue,\nShiji Song, and Gao Huang. 2025. “Does Reinforcement Learning Really\nIncentivize Reasoning Capacity in LLMs Beyond the Base Model?”\narXiv:2504.13837v5. https://arxiv.org/abs/2504.13837v5',70,259,1120,158,25,C.ink,false],['MODEL AND FROZEN EXPERIMENT',70,455,1120,35,22,C.rl,true],['Qwen Team. Qwen3.5-2B-Base.\nhttps://huggingface.co/Qwen/Qwen3.5-2B-Base\nProject: results/final/summary.json; per_question.jsonl; math_l4.json;\nselection.json; formal/report.json. Full citations in speaker notes.',70,507,1120,147,24,C.ink,false]);
 }
 if(i>0){
  shapes.push(['rect',64,676,1152,1,dark?'#38535D':C.line]);
  items.push([i===12?'Sources and methods':'DSAI4207  /  RLVR on MATH Level 4',65,685,1020,22,15,muted,false],[String(i+1).padStart(2,'0'),1170,682,45,26,18,muted,false]);
 }
 for(const [geometry,left,top,width,height,fill] of shapes)s.shapes.add({geometry,position:{left,top,width,height},fill,line:{fill:'none',width:0}});
 for(const [text,left,top,width,height,fontSize,color,bold] of items){const sh=s.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});sh.text=text;sh.text.style={typeface:font,fontSize,color,bold,verticalAlignment:'middle'};}
 for(const [type,config] of charts){for(const ser of config.series){for(const key of ['values','xValues']){if(ser[key])ser[key]=ser[key].map(v=>v===null?v:Number(v.toPrecision(12)));}}const ch=s.charts.add(type,{...config,chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}});applyPresentationChartFont(ch,{fontFamily:font});}
 s.speakerNotes.text=noteSections[i]||'Reference appendix. See SPEAKER_NOTES.md for full sources.';
 console.log('BUILT',i+1,titles[i]);
}

await fs.mkdir(path.join(root,'previews_final'),{recursive:true});
for(const [i,s] of P.slides.items.entries()){
 const png=await P.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(root,'previews_final',`slide_${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 console.log('RENDERED',i+1);
}
const candidate=path.join(root,'.build/candidate.pptx');
await (await PresentationFile.exportPptx(P)).save(candidate);
await fs.writeFile(path.join(root,'.build/inspection.ndjson'),(await P.inspect({kind:'slide,textbox,shape,chart,notes',maxChars:2000000})).ndjson);
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'output/DSAI4207_Group_Presentation.pptx'),pythonExecutable:'/Users/xu./.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],requiredNativeChartOwnerSlides:[2,6,8,9,12],fontPolicy:{basis:'design',families:[font]},materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,receiptPath:path.join(root,'.build/final_validation.json')});
console.log(JSON.stringify(result));
