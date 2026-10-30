"""Rebuild vector research figures from the accompanying audited data snapshot.

Requires reportlab. Run from any working directory with Python 3.12 or later.
The figure data are exported from results/amazon_20k by the report update script.
"""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color, white, black

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'figure_data.json').read_text())
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
BLUE, TEAL, ORANGE = map(HexColor, ['#245882', '#28786F', '#AA542C'])
GRAY, LIGHT, INK = map(HexColor, ['#65717B', '#E3E8EC', '#202830'])
COLORS = [BLUE, TEAL, ORANGE]
MODELS = ['MLP', 'TextCNN', 'BiLSTM']

def new(name, h=210):
    c = canvas.Canvas(str(OUT / (name + '.pdf')), pagesize=(600, h))
    c.setTitle(name.replace('_', ' '))
    return c

def text(c, x, y, s, size=10, color=INK, align='left', bold=False):
    c.setFillColor(color)
    c.setFont('Helvetica-Bold' if bold else 'Helvetica', size)
    getattr(c, {'left':'drawString', 'center':'drawCentredString', 'right':'drawRightString'}[align])(x, y, str(s))

def line(c, x1, y1, x2, y2, color=GRAY, width=0.7, dash=None):
    c.setStrokeColor(color); c.setLineWidth(width); c.setDash(dash or [])
    c.line(x1,y1,x2,y2); c.setDash([])

def box(c, x, y, w, h, labels, color=BLUE):
    c.setFillColor(Color(color.red,color.green,color.blue,alpha=.07))
    c.setStrokeColor(color); c.setLineWidth(.8)
    c.roundRect(x,y,w,h,4,fill=1,stroke=1)
    for i,s in enumerate(labels): text(c,x+w/2,y+h/2+(len(labels)-1)*6-i*12,s,9,align='center')

def arrow(c,x1,y1,x2,y2,color=GRAY):
    import math
    line(c,x1,y1,x2,y2,color)
    a=math.atan2(y2-y1,x2-x1)
    for d in [-.45,.45]: line(c,x2,y2,x2-5*math.cos(a+d),y2-5*math.sin(a+d),color)

def bar(c,x,y,w,h,color):
    c.setFillColor(color); c.rect(x,y,w,h,fill=1,stroke=0)

def axes(c,x,y,w,h,lo,hi,ticks,suffix=''):
    for v in ticks:
        yy=y+(v-lo)/(hi-lo)*h
        line(c,x,yy,x+w,yy,LIGHT,.5)
        text(c,x-6,yy-3,f'{v:g}{suffix}',8,GRAY,align='right')
    line(c,x,y,x+w,y,GRAY)

def finish(c): c.showPage(); c.save()

# Workflow: fitting and model selection do not consume held-out test data.
c=new('experiment_workflow',245)
box(c,8,175,116,51,['Amazon source train','3.6 million reviews'])
box(c,156,175,126,51,['Balanced sampling','20,000 reviews'])
box(c,314,175,126,51,['Shared cleaning','Stratified split'])
arrow(c,124,200,156,200); arrow(c,282,200,314,200)
for y,lab in [(119,['Training 16,000','Fit representations']), (62,['Validation 2,000','Select checkpoints']), (5,['Test 2,000','Final scores only'])]:
    box(c,462,y,130,46,lab)
    arrow(c,440,200,450,y+23); arrow(c,450,y+23,462,y+23)
box(c,164,91,246,58,['Training fitted TF-IDF -> MLP', 'Training fitted tokenizer -> TextCNN / BiLSTM'])
arrow(c,462,142,410,133)
box(c,164,12,246,49,['Two candidates per architecture','Validation loss selects model and epoch'],TEAL)
arrow(c,288,91,288,61); arrow(c,462,85,410,38)
text(c,15,67,'Identical review IDs and labels',10,bold=True)
text(c,15,51,'across all three configurations',9)
text(c,15,24,'Test data never fit vocabulary',9)
text(c,15,11,'or choose hyperparameters',9)
finish(c)

# Architecture diagram is a schematic, not an empirical attribution.
c=new('model_architectures',207)
rows=[('MLP',['Full review','TF-IDF 20,000'],['Dense 256','Dropout'],['Dense 64','Dropout'],['Sigmoid','5,136,769 params']),
      ('TextCNN',['100 token IDs','Embedding 100'],['Conv1D 128','Kernel width 5'],['Max pool','Dense 64'],['Dropout + sigmoid','2,072,449 params']),
      ('BiLSTM',['100 token IDs','Embedding 100'],['BiLSTM','64 per direction'],['Concatenate','Dense 64'],['Dropout + sigmoid','2,092,801 params'])]
for i,(name,a,b,d,e) in enumerate(rows):
    y=145-i*64; color=COLORS[i]
    text(c,3,y+26,name,10,bold=True,color=color)
    for x,w,ls in [(76,117,a),(215,105,b),(342,101,d),(465,131,e)]: box(c,x,y,w,49,ls,color)
    for x1,x2 in [(193,215),(320,342),(443,465)]: arrow(c,x1,y+25,x2,y+25)
text(c,300,2,'BiLSTM masks padding; TextCNN embeds padding without a recurrent mask.',9,GRAY,align='center')
finish(c)

# Two panels document both class allocation and the length restriction.
c=new('dataset_coverage',225)
text(c,150,207,'(a) Balanced partition sizes',11,bold=True,align='center')
text(c,453,207,'(b) Reviews exceeding 100 tokens',11,bold=True,align='center')
axes(c,49,44,213,140,0,16000,[0,4000,8000,12000,16000])
axes(c,356,44,224,140,0,40,[0,10,20,30,40],'%')
for i,key in enumerate(['train','validation','test']):
    n=DATA['splits'][key]['rows']; x=72+i*71; hh=(n/2)/16000*140
    bar(c,x,44,31,hh,BLUE); bar(c,x,44+hh,31,hh,TEAL)
    text(c,x+15,48+2*hh,f'{n:,}',9,align='center')
    text(c,x+15,28,['Train','Validation','Test'][i],8,align='center')
    seq=DATA['sequences'][key]; p=seq['truncated_reviews']/n*100; xx=380+i*72
    bar(c,xx,44,33,p/40*140,ORANGE)
    text(c,xx+16,50+p/40*140,f'{p:.2f}%',9,align='center')
    text(c,xx+16,28,['Train','Validation','Test'][i],8,align='center')
bar(c,65,7,9,9,BLUE); text(c,80,8,'Negative',8)
bar(c,150,7,9,9,TEAL); text(c,165,8,'Positive',8)
text(c,457,8,'Counts: 4,738 / 571 / 593',8,GRAY,align='center')
finish(c)

c=new('metric_comparison',227)
axes(c,45,42,540,147,0,100,[0,20,40,60,80,100],'%')
for j,metric in enumerate(['accuracy','precision','recall','f1']):
    for i,row in enumerate(DATA['metrics']):
        x=75+j*132+i*25; v=row[metric]*100
        bar(c,x,42,20,v/100*147,COLORS[i]); text(c,x+10,48+v/100*147,f'{v:.2f}',8,align='center')
    text(c,110+j*132,24,['Accuracy','Precision','Recall','F1 score'][j],9,align='center')
for i,name in enumerate(MODELS): bar(c,170+i*93,208,9,9,COLORS[i]); text(c,184+i*93,209,name,9)
text(c,315,5,'Precision, recall and F1 refer to the positive class; one fixed test set, n = 2,000.',8,GRAY,align='center')
finish(c)

c=new('confusion_matrices',225)
for i,matrix in enumerate(DATA['confusions']):
    x=35+i*200; top=177; cell=63
    text(c,x+63,207,MODELS[i],11,bold=True,align='center')
    for r in range(2):
        for col in range(2):
            val=matrix[r][col]; y=top-(r+1)*cell
            tint=.12+.8*val/1000
            color=Color(1-tint*(1-BLUE.red),1-tint*(1-BLUE.green),1-tint*(1-BLUE.blue))
            bar(c,x+col*cell,y,cell-1,cell-1,color)
            tc=white if val>600 else INK
            text(c,x+col*cell+cell/2,y+34,str(val),14,tc,align='center',bold=True)
            text(c,x+col*cell+cell/2,y+17,f'{val/10:.1f}%',9,tc,align='center')
    for col in range(2): text(c,x+col*cell+cell/2,37,['Negative','Positive'][col],8,align='center')
    text(c,x+63,21,'Predicted label',9,align='center')
    c.saveState(); c.translate(x-13,113); c.rotate(90); text(c,0,0,'Actual: negative / positive',8,align='center'); c.restoreState()
text(c,300,4,'Each actual-class row has 1,000 reviews. Percentages are row-normalized.',8,GRAY,align='center')
finish(c)

c=new('training_curves',211)
for i,hist in enumerate(DATA['histories']):
    x=39+i*200; y=41; w=145; h=126
    text(c,x+w/2,192,MODELS[i],11,bold=True,align='center')
    axes(c,x,y,w,h,0,.8,[0,.2,.4,.6,.8])
    n=len(hist)
    for field,color,dash in [('loss',BLUE,None),('val_loss',ORANGE,[4,2])]:
        pts=[(x+(int(row['epoch'])-1)/(max(4,n)-1)*w,y+float(row[field])/.8*h) for row in hist]
        for a,b in zip(pts,pts[1:]): line(c,*a,*b,color,1.5,dash)
        for xx,yy in pts:
            c.setFillColor(color); c.circle(xx,yy,2.4,fill=1,stroke=0)
    best=DATA['selected'][i]['best_epoch']; xx=x+(best-1)/3*w
    line(c,xx,y,xx,y+h,GRAY,.6,[2,3])
    for e in range(1,5): text(c,x+(e-1)/3*w,y-13,str(e),8,align='center')
    text(c,x+w/2,14,'Epoch',9,align='center')
line(c,171,4,191,4,BLUE,1.5); text(c,197,1,'Training loss',8)
line(c,300,4,320,4,ORANGE,1.5,[4,2]); text(c,326,1,'Validation loss',8)
text(c,300,176,'Binary cross-entropy; dotted vertical line marks the retained epoch',8,GRAY,align='center')
finish(c)

c=new('error_overlap',226)
groups=DATA['error_regions']; labels=['MLP only','CNN only','LSTM only','MLP + CNN','MLP + LSTM','CNN + LSTM','All three']
axes(c,45,59,540,127,0,140,[0,35,70,105,140])
for i,(label,v) in enumerate(zip(labels,groups)):
    xx=67+i*75; col=BLUE if i<3 else TEAL if i<6 else ORANGE
    bar(c,xx,59,40,v/140*127,col); text(c,xx+20,64+v/140*127,str(v),10,align='center')
    text(c,xx+20,43,label,8,align='center')
text(c,300,207,'Exact membership of the union of model error sets',11,align='center',bold=True)
text(c,300,23,'470 distinct reviews contain at least one error; 103 are missed by all three.',9,align='center')
text(c,300,7,'Pairwise bars exclude the all-three intersection. Correct on all models: 1,530 reviews.',8,GRAY,align='center')
finish(c)

c=new('resource_comparison',245)
text(c,154,224,'(a) Selected training stage duration',11,bold=True,align='center')
text(c,458,224,'(b) Training feature storage',11,bold=True,align='center')
axes(c,49,52,227,145,0,240,[0,60,120,180,240])
for i,sec in enumerate(DATA['selected_stage_seconds']):
    x=70+i*74; bar(c,x,52,36,sec/240*145,COLORS[i]); text(c,x+18,58+sec/240*145,f'{sec:.2f} s',9,align='center'); text(c,x+18,35,MODELS[i],9,align='center')
text(c,44,207,'Seconds',8,GRAY)
# Storage is log-scaled, since a linear axis would obscure sparse storage.
import math
xx=361; yy=52; ww=215; hh=145
for p in range(5):
    y=yy+p/4*hh; line(c,xx,y,xx+ww,y,LIGHT,.5); text(c,xx-6,y-3,str(10**p),8,GRAY,align='right')
mem=[DATA['csr_bytes']/1e6,16000*20000*4/1e6,DATA['dense64_bytes']/1e6]
for i,v in enumerate(mem):
    x=382+i*70; bar(c,x,yy,34,math.log10(v)/4*hh,[TEAL,GRAY,ORANGE][i]); text(c,x+17,yy+math.log10(v)/4*hh+6,f'{v:.2f}',9,align='center'); text(c,x+17,35,['CSR f32','Dense f32','Dense f64'][i],8,align='center')
text(c,355,207,'MB (decimal), logarithmic axis',8,GRAY)
text(c,300,17,'Stage times include process overhead and differ in epochs run; these are not normalized speed benchmarks.',8,GRAY,align='center')
text(c,300,3,'Dense sizes are calculated baselines. Storage bars do not represent measured total process memory.',8,GRAY,align='center')
finish(c)
print('Built eight vector figures from figure_data.json')
