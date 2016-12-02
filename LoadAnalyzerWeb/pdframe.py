# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#!/home/gianluca/Apps/anaconda/bin/python

from lxml import etree
import pandas as pd
import numpy
import bisect
import datetime
try:
	from geopy.distance import great_circle
except ImportError:
	from geopy.distance import GreatCircleDistance as great_circle
#import matplotlib.pyplot as plt
#from scipy import stats
import math
#import mkl




# <codecell>


# <markdowncell>

# Read and parse gpx file

# <codecell>


def savitzky_golay(y, window_size, order, deriv=0, rate=1):

	import numpy as np
	from math import factorial
	
   

	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
   
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')

# Compute de dataframe from the GPX file
def ReadGpxFile( gpxfilename ):

	gpxfile = etree.parse(gpxfilename)
	root  = gpxfile.getroot()

	try:
		ns = '{'+root.nsmap[None]+'}'
	except KeyError:
		ns = ''

	# <markdowncell>
	# Putting the point attribute in a pandas DataFrame
	# <codecell>

	datalist = []
	for tr in gpxfile.getiterator(ns+'trk'):
		for seg in tr.getiterator(ns+'trkseg'):
			for p in seg.getiterator(ns+'trkpt'):
				lat, lon = float(p.attrib['lat']), float(p.attrib['lon'])                            
				try:
					accX, accY,  accZ = p.find(ns+'accX'),p.find(ns+'accY'), p.find(ns+'accZ')
					accX = [float(x) for x in accX.text.split(',')]
					accY = [float(x) for x in accY.text.split(',')]
					accZ = [float(x) for x in accZ.text.split(',')]                   
				except AttributeError:
					accX,accY,accZ = [], [], []
					
#                try:
#                    magX, magY,  magZ = p.find(ns+'magX'), p.find(ns+'magY'), p.find(ns+'magZ')
#                    magX = [float(x) for x in magX.text.split(',')]
#                    magY = [float(x) for x in magY.text.split(',')]
#                    magZ = [float(x) for x in magZ.text.split(',')]
#                except AttributeError:
#                    magX, magY, magZ = [], [], []
					
				ptime = p.find(ns+'time')
				
				try:
					ptimetext = ptime.text
				except AttributeError:
					ptimetext = ''
				
				try:
					gpsaccuracy = p.find(ns+'gpsaccuracy')
					gpsaccuracy = float(gpsaccuracy.text)
				except AttributeError:
					gpsaccuracy = None

				datalist.append([lon, lat, ptimetext, accX, accY, accZ, gpsaccuracy])
	gpxDataFrame = pd.DataFrame(data = datalist, columns=['Longitude', 'Latitude', 'Time', 'accX', 'accY', 'accZ', 'GpsAccuracy'])
	
	gpxDataFrame = AggregateLonLat(gpxDataFrame)    
	
	return gpxDataFrame

# reindex the pandas dataframe with consecutive indices starting from 0    
def ReindexDataFrame( dataframe ):
	newindices = {}
	for i in range(dataframe.shape[0]):
		newindices[i] = i
	dataframe.index = newindices
	return dataframe  
  
# Add to the dataframe in input colums Seconds, Distance,...
# deltasec is used for compute the speed that is averaged
# on deltasec seconds
def AddMoreColumns( gpxDataFrame, deltasec = 5 ):

	# <markdowncell>
	# Creating an additional column Seconds (sec.millisec) from the beginning
	
	# <codecell>
	
	gpxDataFrame = ReindexDataFrame(gpxDataFrame)
	
	#irow return the row of the index in input
	startingtime = gpxDataFrame.Time[0]
	try:
		startingtime = datetime.datetime.strptime(startingtime[:-5], "%Y-%m-%dT%H:%M:%S.%f")
	except ValueError:
		try:
			startingtime = datetime.datetime.strptime(startingtime[:19], "%Y-%m-%dT%H:%M:%S")
		except ValueError:
			startingtime = datetime.datetime(1970,1,1,0,0,0)
	
	# <codecell>

	deltasecs = []
	for x in gpxDataFrame.Time:
		try:
			newtime = datetime.datetime.strptime(x[:-5], "%Y-%m-%dT%H:%M:%S.%f")
		except ValueError:
			try:
				newtime = datetime.datetime.strptime(x[:19], "%Y-%m-%dT%H:%M:%S")
			except ValueError:
				newtime = datetime.datetime(1970,1,1,0,0,0)
		deltasecs.append((newtime-startingtime).total_seconds())
	
	# <codecell>
	
	gpxDataFrame['Seconds'] = deltasecs

	# <markdowncell>
	
	# Creating distance column and load from the previous point
	
	# <codecell>
	
	distances = [0]
	loads = [0]
	
	for p in gpxDataFrame.index[1:]:
		dist = great_circle((gpxDataFrame.Latitude[p], gpxDataFrame.Longitude[p]), \
							(gpxDataFrame.Latitude[p-1], gpxDataFrame.Longitude[p-1])).meters
		newload = math.sqrt( (pd.Series(gpxDataFrame.accX[p]).mean() - pd.Series(gpxDataFrame.accX[p-1]).mean())**2 + \
							 (pd.Series(gpxDataFrame.accY[p]).mean() - pd.Series(gpxDataFrame.accY[p-1]).mean())**2 + \
							 (pd.Series(gpxDataFrame.accZ[p]).mean() - pd.Series(gpxDataFrame.accZ[p-1]).mean())**2 \
							)/10
		distances.append( dist )
		loads.append( newload )
		
	
	# <codecell>
	
	gpxDataFrame['Distance'] = distances
	gpxDataFrame['TotDistance'] = gpxDataFrame.Distance.cumsum()
	gpxDataFrame['Load'] = loads

	# <markdowncell>
	
	# Setting the speed: it is taken every <tt>deltasec</tt> seconds
	
	# <codecell>
	
	first = 0
	speeds, totloads = [], []
	for p in gpxDataFrame.index:
		if p == gpxDataFrame.index[-1] or gpxDataFrame.Seconds[p+1] -  gpxDataFrame.Seconds[first] > deltasec:
			newspeed = gpxDataFrame.Distance[first:p+1].sum()/(gpxDataFrame.Seconds[p] - gpxDataFrame.Seconds[first])
			speeds = speeds + [newspeed*3.6]*(p-first+1)
			load = gpxDataFrame.Load[first:p+1].sum()
			totloads = totloads + [load]*(p-first+1)
			dist, load, first = 0, 0, p+1
		

	# <codecell>
	
	gpxDataFrame['Speed'] = speeds    
	gpxDataFrame['TotLoad'] = totloads
	
	return gpxDataFrame


def AddSimpleLoadCoumn( gpxDataFrame ):
	simpleload = []
	gpxDataFrame = ReindexDataFrame(gpxDataFrame)
	for p in gpxDataFrame.index:
		simpleload.append(math.sqrt(pd.Series(gpxDataFrame.accX[p]).mean()**2\
			+ pd.Series(gpxDataFrame.accY[p]).mean()**2\
			+ pd.Series(gpxDataFrame.accZ[p]).mean()**2))
	gpxDataFrame['SimpleLoad'] = simpleload
	return gpxDataFrame

def AddVarLoadColumn( gpxDataFrame ):
	# idea: the points with high variance on accX, accY, accZ
	# with more load
	varload = []
	gpxDataFrame = ReindexDataFrame(gpxDataFrame)
	for p in gpxDataFrame.index:
		if pd.isnull(pd.Series(gpxDataFrame.accX[p]).var()):
			varx = 0
		else:
			varx = pd.Series(gpxDataFrame.accX[p]).var()
		if pd.isnull(pd.Series(gpxDataFrame.accY[p]).var()):
			vary = 0
		else:
			vary = pd.Series(gpxDataFrame.accY[p]).var()
		if pd.isnull(pd.Series(gpxDataFrame.accY[p]).var()):
			varz = 0
		else:
			varz = pd.Series(gpxDataFrame.accY[p]).var()
		
		varload.append((varx+vary+varz)/3)
	gpxDataFrame['VarLoad'] = varload
	return gpxDataFrame
	
	
def CreateCsvFile( gpxDataFrame, filename ):
	# header    
	gpxstring = '"Longitude","Latitude","Speed","VarLoad","VarLoadSmooth","Load","TotLoad"\n'

	gpxDataFrame = gpxDataFrame.fillna(0)
	
	for p in gpxDataFrame.index:
		gpxstring += str(gpxDataFrame.Longitude[p])\
			+','+str(gpxDataFrame.Latitude[p])\
			+','+str(gpxDataFrame.Speed[p])\
			+','+str(gpxDataFrame.VarLoad[p])\
			+','+str(gpxDataFrame.VarLoadSmooth[p])\
			+','+str(gpxDataFrame.Load[p])\
			+','+str(gpxDataFrame.TotLoad[p])\
			+'\n'
	# tailer    
	gpxstring += '' 
   
   
	fileout = open(filename, 'w')
	fileout.write(gpxstring)    
	fileout.close()    
	
def GetDataFrame( gpxfilename, morecols=True):
	df = ReadGpxFile(gpxfilename)
	# filter based on gps accuracy
	#df = df[df.GpsAccuracy <= 6]
	if morecols:
		df = AddMoreColumns( df, deltasec = 15 )
		#df = (df[df.Seconds.max() - df.Seconds > 10])[df.Seconds > 10]
		df = AddVarLoadColumn( df )
		#df['NormLoad'] = df.VarLoad/df.VarLoad.max()
	return df
	
def FilterPoints( vlayer, stringexpr ):
	# example stringexpr = 'Load > 0.1'
	exp=QgsExpression(stringexpr)
	for f in list(vlayer.getFeatures()):
		value = exp.evaluate(f)
		if value > 0:
			vlayer.setSelectedFeatures([f.id()])

def DefineColors(trackDataFrames, attribute = 'VarLoadSmooth'):
	if len(trackDataFrames) > 0:
		mins = [ trackDataFrames[i].VarLoadSmooth.min() for i in range(len(trackDataFrames))]
		maxs = [ trackDataFrames[i].VarLoadSmooth.max() for i in range(len(trackDataFrames))]
		tmin, tmax = min(mins), max(maxs)
		
		segments = numpy.arange(tmin, tmax, (tmax-tmin)/5)
		colors = ['lightgreen', 'cyan', 'yellow', 'red', 'black']
		
		for trackDF in trackDataFrames:
			colorlist = [None]
			for i in trackDF.index[1:]:
				colorlist.append(colors[bisect.bisect_right(segments, trackDF.ix[i][attribute])-1])
			trackDF['Color'] = colorlist
	return trackDataFrames



#########
# The list segments contains info about how to color the points
# Are used len(segents)-1 colors, the first color is assigned
# to points with attribute between segments[0] and segments[1],
# the second to points with attribute between segments[1] and segments[2]
# and so on
def GraduadedVectorDrawing( layer, attribute, segments ):
	myRangeList = []
	
	myOpacity = 1
	
	myColour = QColor('#0000FF') # blue
	mySymbol1 = QgsSymbolV2.defaultSymbol(layer.geometryType())
	mySymbol1.setColor(myColour)
	mySymbol1.setAlpha(0.2)
	myRange1 = QgsRendererRangeV2(segments[0], segments[1], mySymbol1, '')
	myRangeList.append(myRange1)
	
	myColour = QColor('#00FF00') # green 
	mySymbol2 = QgsSymbolV2.defaultSymbol(layer.geometryType())
	mySymbol2.setColor(myColour)
	mySymbol2.setAlpha(myOpacity)
	myRange2 = QgsRendererRangeV2(segments[1], segments[2], mySymbol2, '')
	myRangeList.append(myRange2)
	
	myColour = QColor('#FF0000') # red 
	mySymbol3 = QgsSymbolV2.defaultSymbol(layer.geometryType())
	mySymbol3.setColor(myColour)
	mySymbol3.setAlpha(myOpacity)
	myRange3 = QgsRendererRangeV2(segments[2], segments[3], mySymbol3, '')
	myRangeList.append(myRange3)
	
	myColour = QColor('#000000') # black
	mySymbol4 = QgsSymbolV2.defaultSymbol(layer.geometryType())
	mySymbol4.setColor(myColour)
	mySymbol4.setAlpha(myOpacity)
	myRange4 = QgsRendererRangeV2(segments[3], segments[4], mySymbol4, '')
	myRangeList.append(myRange4)
	
	
	myRenderer = QgsGraduatedSymbolRendererV2('', myRangeList)
	#myRenderer.setMode(QgsGraduatedSymbolRendererV2.EqualInterval)
	myRenderer.setClassAttribute(attribute)
	
	layer.setRendererV2(myRenderer)
	#return layer
	QgsMapLayerRegistry.instance().addMapLayer(layer)
	iface.mapCanvas().refresh()


#########################################
def f(x):
	lr =list(x)
	nlr = [x for sl in lr for x in sl]
	return tuple(nlr)
	
# In the case points are repeated (sequence of points with same coordinates)
# use this function for merge row by gouping. Function f creates the list
# of accX, accY, accZ
def AggregateLonLat( tr ):
	# a new column that join latitude and longitude
	tr['LatLonSig'] = [ str(tr.Longitude[i])+str(tr.Latitude[i]) for i in range(tr.shape[0])]    
	# the colum block contains progressive integer, rows with the same
	# index have same Longitude and Latitude and are consecutive
	tr['Block'] = (tr.LatLonSig.shift(1) != tr.LatLonSig).astype(int).cumsum()
	
	# Now we can group and aggregate by using Block
	group = tr.groupby([tr.Block],  sort = False)    
	gtr = group.agg({ 'Longitude': 'last', 'Latitude': 'last',\
		'accX': f, 'accY': f, 'accZ': f,\
		'GpsAccuracy': 'min', 'Time': 'min', 'LatLonSig': 'last', 'Block': 'last' })
 
	laccX, laccY, laccZ = [ list(x) for x in  gtr.accX],\
		[ list(x) for x in  gtr.accY],\
		[ list(x) for x in  gtr.accZ]
 
	gtr = ReindexDataFrame(gtr)
	outdf = pd.DataFrame()
	outdf['Longitude'] = gtr.Longitude
	outdf['Latitude'] = gtr.Latitude
	outdf['Time'] = gtr.Time
	outdf['GpsAccuracy'] = gtr.GpsAccuracy
	outdf['accX'], outdf['accY'], outdf['accZ'] = laccX, laccY, laccZ
	return outdf


# Considers sequences of points of tot length "length" and sort
# by property "prop"
def SmoothByLength( tr, length = 10, prop = 'VarLoad', newprop = 'VarLoadSmooth' ):
	# for all points i a[i] = k if dist(i...k-1) < lenght and
	# dist(i...k) >= length
	a = []
	for i in tr.index:
		k = i
		while k < tr.index[-1] and tr.Distance[i:k+1].sum() < length:
			k += 1
		a.append((i, k+1, tr.Distance[i:k+1].sum(), tr[prop][i:k+1].sum() ))
		
	b = pd.Series(0.0, index=tr.index)
	
	for i, k, dist, load in a:
		j = i;
		while j < k:
		   b[j] = max(b[j], load)
		   j += 1
		   
	tr[newprop] = b
	return tr
		
		


foldername = '/home/gianluca/Dropbox/work_in_progress/LoadAnalyzer/tracce_gpx/'
#gpxfilename = '/MLA-20160307-135408.gpx'
## Tempio
#gpxfilename = '/20160311 ciana tempio/MLA-20160311-172153.gpx'
#gpxfilename = '/20160315 ciana - tempio/MLA-20160315-084425.gpx'
## Ciana
#gpxfilename = '/20160311 ciana tempio/MLA-20160311-163929.gpx'
#gpxfilename = '/20160315 ciana - tempio/MLA-20160315-081359.gpx'
#gpxfilename = '/20160315 ciana - tempio/MLA-20160315-082603.gpx'
#gpxfilename =  '/20160408 ciana - tempio/MLA-20160408-173540.gpx'
#gpxfilename = '/20160412 ciana - tempio/MLA-20160412-080807.gpx'
## Centro
#gpxfilename = '/20160315 ciana - tempio/MLA-20160315-085015.gpx'
## Tronchetto
#gpxfilename = '20160313 vallerotonda/MLA-20160312-144749.gpx'
##Cavallo Bianco - Fiora
gpxfilename = '/20160318 cavallo bianco - fiora/MLA-20160318-161437.gpx'
#gpxfilename = '/20160318 cavallo bianco - fiora/MLA-20160318-164715.gpx'
## Forcola
## to be aggregated
##gpxfilename = '/20160304-forcola-ciana-tempio/MLA-20160304-103859.gpx'
# Vallerotonda
#127
#gpxfilename = '/20160326 vallerotonda/MLA-20160326-125200.gpx'
## Sperlonga
#gpxfilename = '20160328 sperlonga/MLA-20160328-124243.gpx'




if False:
	tr = GetDataFrame( foldername+gpxfilename )
	y = savitzky_golay(tr.VarLoad.tolist(), 9, 2)
	tr['VarLoadSmooth'] = y
	

if False:
	tr = GetDataFrame( foldername+gpxfilename )

	tr = SmoothByLength(tr)
	
	csvfilename = foldername + gpxfilename+'.csv'
	#CreateCsvFile( tr[tr.VarLoad >= 0.20*tr.VarLoad.max()], csvfilename)
	CreateCsvFile( tr, csvfilename)
		
	uri = csvfilename + '?crs=epsg:4326&delimiter=%s&xField=%s&yField=%s' % (",", "Longitude", "Latitude")
	
	layer = QgsVectorLayer(uri, "TR VarLoad", "delimitedtext")
	crs = QgsCoordinateReferenceSystem()
	crs.createFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
	layer.setCrs(crs)
	QgsMapLayerRegistry.instance().addMapLayer(layer)
	
	GraduadedVectorDrawing( layer, attribute = 'VarLoadSmooth', segments = [i*1.0*(tr.VarLoadSmooth.max()-tr.VarLoadSmooth.min())/4 for i in [0, 1.0, 2.0, 3.0, 4.0] ] )
	#GraduadedVectorDrawing( layer, attribute = 'VarLoad', segments = [tr.VarLoad.quantile(x/4) for x in [0, 1.0, 2.0, 3.0, 4.0]] )

	

if False:
	stringexpr = 'Load >=70'
	exp=QgsExpression(stringexpr)
	selectedfeatures = []
	for f in layer.getFeatures():
		value = exp.evaluate(f)
		if value > 0:
			selectedfeatures.append(f.id())
			
	layer.setSelectedFeatures(selectedfeatures)

	
	#layer=iface.activeLayer()





if False:
	if False: # in the case the accX,... values are not aggregated in list 
		tr = GetDataFrame( foldername+gpxfilename, morecols=False )
		tr = AggregateLonLat( tr )
		tr = AddMoreColumns( tr, deltasec = 15 )
		tr = (tr[tr.Seconds.max() - tr.Seconds > 10])[tr.Seconds > 10]
		tr = AddVarLoadColumn( tr )
	else:
		tr = GetDataFrame( foldername+gpxfilename)




