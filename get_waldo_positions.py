import xml.etree.ElementTree
import glob
import json
import re

ANNOTATIONS_PATH = './images/annotations/*'

# converts bounding boxes of waldo (generated from labelimg) into dictionary format
def main():
	d = {}
	files = glob.glob(ANNOTATIONS_PATH)
	for f in files:
		root = xml.etree.ElementTree.parse(f).getroot()
		points = []
		for bounding_box in root.iter('bndbox'):
			p = []
			for point in bounding_box:
				# xmin, ymin, xmax, ymax
				p.append(float(point.text))
			points.append(p)
		key = re.match('^.*(waldo[0-9]*).*$', f).group(1)
		d[key] = points
	with open('waldo_locations.json', 'w+') as f:
		f.write(json.dumps(d))

if __name__ == '__main__':
	main()