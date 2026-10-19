"""Draw the VOXPrint AI Canvas icon: assets/voxprint-canvas.png (1024 px) and assets/voxprint-canvas.ico.

Run from the repository root: ``python tools/make_icon.py``. Needs Pillow. The .ico holds 16, 24, 32, 48,
64, 128 and 256 px frames; 16-32 px are drawn on their own pixel grid with less detail.
"""
from pathlib import Path

from PIL import Image, ImageDraw
B=(0x4d,0xa3,0xff,255); SH=(0x2b,0x6c,0xc4,255); SEA=(0x17,0x3f,0x7a,255); SEA2=(0x3a,0x78,0xc8,255)
SKY_T=(0x1f,0x2a,0x52); SKY_B=(0xf2,0x9a,0x62); SUN=(0xff,0xd2,0x7a,255)
HZ=0.78
ROCK=[(0,HZ),(0.10,HZ),(0.12,0.74),(0.135,0.52),(0.15,0.36),(0.175,0.29),(0.20,0.27),(0.26,0.30),(0.31,0.28),(0.36,0.32),(0.44,0.27),(0.52,0.22),(0.57,0.235),(0.63,0.37),(0.69,0.51),(0.72,0.56),(0.83,0.575),(0.86,0.645),(0.90,0.68),(0.97,0.69),(1.0,0.72),(1.0,HZ)]
SHADOW=[(0.10,HZ),(0.12,0.74),(0.135,0.52),(0.15,0.36),(0.175,0.29),(0.20,0.27),(0.24,0.29),(0.22,0.45),(0.20,0.62),(0.19,HZ)]
def tile(S):
    img=Image.new("RGBA",(S,S),(0,0,0,0)); m=round(S*.04); r=round((S-2*m)*.18)
    g=Image.new("RGBA",(S,S)); gd=ImageDraw.Draw(g)
    for y in range(S):
        t=y/S; gd.line([(0,y),(S,y)],fill=tuple(int(a+(b-a)*t) for a,b in zip((0x1b,0x1b,0x23),(0x2c,0x2c,0x38)))+(255,))
    mk=Image.new("L",(S,S),0); ImageDraw.Draw(mk).rounded_rectangle([m,m,S-m-1,S-m-1],r,fill=255)
    img.paste(g,(0,0),mk); return img
def scene(S,box,w,rr,detail=True):
    img=tile(S); x0,y0,x1,y1=box; ix0,iy0,ix1,iy1=x0+w,y0+w,x1-w,y1-w
    iw,ih=ix1-ix0,iy1-iy0; P=lambda px,py:(ix0+px*iw,iy0+py*ih)
    sc=Image.new("RGBA",(S,S),(0,0,0,0)); d=ImageDraw.Draw(sc)
    hz=iy0+HZ*ih
    for y in range(int(iy0),int(hz)+1):
        t=(y-iy0)/(hz-iy0); d.line([(ix0,y),(ix1,y)],fill=tuple(int(a+(b-a)*t**1.6) for a,b in zip(SKY_T,SKY_B))+(255,))
    if detail:
        cx,cy=P(0.78,0.36); R=0.09*iw; d.ellipse([cx-R,cy-R,cx+R,cy+R],fill=SUN)
    d.rectangle([ix0,hz,ix1,iy1],fill=SEA)
    if detail:
        for k,(a,b) in enumerate(((0.30,0.62),(0.66,0.92),(0.12,0.40))):
            yy=hz+(0.30+0.25*k)*(iy1-hz); d.rounded_rectangle([ix0+a*iw,yy,ix0+b*iw,yy+0.022*ih],int(0.011*ih),fill=SEA2)
    d.polygon([P(*p) for p in ROCK],fill=B)
    d.polygon([P(*p) for p in SHADOW],fill=SH)
    inner=Image.new("L",(S,S),0); ImageDraw.Draw(inner).rounded_rectangle([ix0,iy0,ix1,iy1],max(1,rr-w),fill=255)
    img.paste(sc,(0,0),inner)
    fm=Image.new("L",(S,S),0); fd=ImageDraw.Draw(fm); fd.rounded_rectangle(box,rr,fill=255); fd.rounded_rectangle([ix0,iy0,ix1,iy1],max(1,rr-w),fill=0)
    img.paste(Image.new("RGBA",(S,S),B),(0,0),fm); return img


def main(folder: Path = Path(__file__).resolve().parents[1] / "assets") -> None:
    S = 4096
    master = scene(S, [int(S*.16), int(S*.24), int(S*.84), int(S*.76)], int(S*.05), int(S*.07)).resize((1024, 1024), Image.LANCZOS)
    folder.mkdir(parents=True, exist_ok=True)
    master.save(folder / "voxprint-canvas.png")
    fr = {n: master.resize((n, n), Image.LANCZOS) for n in (48, 64, 128, 256)}
    for n, box, w in ((16, (1, 3, 14, 12), 1), (24, (2, 5, 21, 18), 2), (32, (3, 7, 28, 24), 2)):
        k = 16; Sb = n*k; bx = [box[0]*k, box[1]*k, (box[2]+1)*k-1, (box[3]+1)*k-1]
        fr[n] = scene(Sb, bx, w*k, int(1.5*k), detail=n >= 32).resize((n, n), Image.BOX)
    o = (256, 128, 64, 48, 32, 24, 16)
    fr[256].save(folder / "voxprint-canvas.ico", format="ICO", append_images=[fr[n] for n in o[1:]], sizes=[(n, n) for n in o])


if __name__ == "__main__":
    main()
