"""在广告页正在播放时手动执行，抓快照诊断。"""
import uiautomator2 as u2
d = u2.connect()
print("当前 app:", d.app_current())
print("窗口尺寸:", d.window_size())
d.screenshot('ad_playing.png')
xml = d.dump_hierarchy()
open('ad_playing.xml','w').write(xml)
print(f"dumped: ad_playing.png + ad_playing.xml ({len(xml)} bytes)")
# 看顶层 5 个 package
import re
pkgs = set(re.findall(r'package="([^"]+)"', xml))
print("出现的 package:", pkgs)
