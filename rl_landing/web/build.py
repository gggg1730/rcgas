"""shell.html에 sim.js·app.js를 넣어 한 파일(dist/roadster.html)로 만든다. policies.json은 옆에 따로 올림"""
import os
here = os.path.dirname(os.path.abspath(__file__))
read = lambda n: open(os.path.join(here, n), encoding="utf-8").read()
html = read("shell.html").replace("/*SIM_JS*/", read("sim.js")).replace("/*APP_JS*/", read("app.js"))
os.makedirs(os.path.join(here, "dist"), exist_ok=True)
open(os.path.join(here, "dist", "roadster.html"), "w", encoding="utf-8").write(html)   # Artifact용 (껍데기는 게시 때 붙음)
# GitHub Pages 등 일반 호스팅용: 문서 껍데기를 직접 붙인 단독 실행 버전
SKELETON = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>[hidden]{display:none!important}body{margin:0}</style>\n</head>\n<body>\n')
open(os.path.join(here, "dist", "index.html"), "w", encoding="utf-8").write(SKELETON + html + "\n</body>\n</html>\n")
print("dist/roadster.html, dist/index.html", len(html) // 1024, "KB")
