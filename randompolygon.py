from PIL import Image, ImageDraw
import random
import uuid

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

def cross(p1, p2, q1, q2):
	# line 1: ax + by + c = 0
	a = p2.y - p1.y
	b = p1.x - p2.x
	c = p2.x * p1.y - p1.x * p2.y
	# line 2 = dx + ey + f = 0
	d = q2.y - q1.y
	e = q1.x - q2.x
	f = q2.x * q1.y - q1.x * q2.y
	# cross point of two line (x,y)
	# aex + bey + ce = 0
	# bdx + bey + bf = 0
	# (ae-bd)x + (ce-bf) = 0
	# x = - (ce-bf) / (ae-bd)
	# adx + bdy + cd = 0
	# adx + aey + af = 0
	# (bd-ae)y + (cd-af) = 0
	# y =  - (cd-af) / (bd-ae)
	x = (b*f - c*e) / (a*e - b*d)
	y = (a*f - c*d) / (b*d - a*e)
	if (p1.x < x < p2.x or p2.x < x < p1.x) and \
		(p1.y < y < p2.y or p2.y < y < p1.y) and \
		(q1.x < x < q2.x or q2.x < x < q1.x) and \
		(q1.y < y < q2.y or q2.y < y < q1.y):
		return Point(x,y)
	return None
		
class Edge:
	def __init__(self, p, q):
		self.p = p
		self.q = q

def ccw(p1, p2, p3):
		return (p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x)

class Polygon:
	def __init__(self, ps):
		minidx = None
		for i in range(len(ps)):
			if minidx == None or ps[minidx].x > ps[i].x:
				minidx = i
		pivot = ps[minidx]
		newps = sorted(ps[:minidx] + ps[minidx+1:], key=lambda p:(p.y-pivot.y)/(p.x-pivot.x) if p.x != pivot.x else float('int'))
		realps = [pivot, newps[0]]
		for i in range(1, len(newps)):
			while ccw(realps[-2], realps[-1], newps[i]) < 0:
				realps.pop()
			realps.append(newps[i])
		self.ps = realps
	def contains(self, p):
		global W
		crosscnt = 0
		for i in range(len(self.ps)):
			if cross(p, Point(W+1, p.y), self.ps[i], self.ps[(i+1)%n]) != None:
				crosscnt = crosscnt + 1
		return crosscnt % 2 == 1

def generaterandomimage(filename, oraclename):
	W = 500
	H = 500
	bgcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
	polycolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
	img = Image.new("RGB", (W, H), bgcolor)
	imgdraw = ImageDraw.Draw(img)
	n = random.randint(3,10)
	newpoly = Polygon([Point(random.randint(0,H-1),random.randint(0,W-1)) for _ in range(n)])
	imgdraw.polygon([(p.x, p.y) for p in newpoly.ps], fill=polycolor, outline=polycolor)
	img.save(filename)
	with open(oraclename, 'at') as f:
		f.write(filename+','+str(n))
		for p in newpoly.ps:
			f.write(','+str(p.x)+' '+str(p.y))
		f.write('\n')

for i in range(10):
	generaterandomimage('dataset/'+str(uuid.uuid1())+'.png', 'oracle.csv')
