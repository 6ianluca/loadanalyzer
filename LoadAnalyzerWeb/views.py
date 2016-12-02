from django.http import HttpResponse, HttpResponseRedirect
import pandas as pd
import django
import datetime
import json
from django.template.loader import get_template
from django.template import Context
from django.shortcuts import render
from django.core.urlresolvers import reverse

from django.views.generic import TemplateView

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from django.shortcuts import render

from django.template import RequestContext

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import forms

from . import pdframe

# globals
default_ne, default_sw = [50, 15], [13, 0]

def DefinesTracks4Web(request):
	trackDataFrames = [ pd.read_json(x) for x in request.session['trackDataFrames'] ]
	trackDataFrames = pdframe.DefineColors(trackDataFrames)
	request.session['trackDataFrames'] = [x.to_json() for x in trackDataFrames]
	trackDF4Web = []
	colorsBound = {} # colorsBound['color'] = maximum load painted with 'color'
	for tdf in trackDataFrames:
		speed = tdf.Speed.ewm(span=10).mean()
		tdf4w = tdf[['Longitude', 'Latitude', 'Color', 'TotDistance', 'VarLoadSmooth', 'Seconds' ]]
		colors = list(set(list(tdf4w.Color)))
		for col in colors:
			if col != None:
				if col in colorsBound:
					colorsBound[col] = max(colorsBound[col], tdf4w.VarLoadSmooth[tdf4w.Color == col].max())
				else:
					colorsBound[col] = tdf4w.VarLoadSmooth[tdf4w.Color == col].max()
		tdf4w['Speed'] = list(speed)
		trackDF4Web.append(tdf4w.to_json())

	return trackDF4Web, colorsBound

def manageselected(request):
	request.session['selectedTrack'], request.session['endindex'] = -1, -1
	
	if request.method == 'POST':
		listofchoices = json.loads(request.session['listofchoices'])
		selTrackForm = forms.FileListForm(request.POST, choices=listofchoices)
		request.session['selectedTrack'] = selTrackForm.data['filelist']
		selectedTrack = request.session['selectedTrack']
		trackDataFrames = [ pd.read_json(x) for x in request.session['trackDataFrames'] ]
		
		if( 'deletebutton' in request.POST):
			del(trackDataFrames[int(selectedTrack)])
			del(listofchoices[int(selectedTrack)])
			temp = []
			for _,x in listofchoices:
			    temp.append( (str(len(temp)),x) )
			request.session['listofchoices'] = json.dumps(temp)
			request.session['selectedTrack'] = -1
			request.session['trackDataFrames'] = [x.to_json() for x in trackDataFrames]
			
			sw, ne = [], []
			if len(trackDataFrames) > 0:
				sw = [ min([trackDataFrames[i].Latitude.min() for i in range(len(trackDataFrames))]),\
					min([trackDataFrames[i].Longitude.min() for i in range(len(trackDataFrames))]) ]
				ne = [ max([trackDataFrames[i].Latitude.max() for i in range(len(trackDataFrames))]),\
					max([trackDataFrames[i].Longitude.max() for i in range(len(trackDataFrames))]) ]
			request.session['sw'], request.session['ne'] = sw, ne 
			
			return HttpResponseRedirect(reverse('home'))
		elif 'zoombutton' in request.POST: #zoom on the
			sw = [ trackDataFrames[int(selectedTrack)].Latitude.min(), trackDataFrames[int(selectedTrack)].Longitude.min()]
			ne = [ trackDataFrames[int(selectedTrack)].Latitude.max(), trackDataFrames[int(selectedTrack)].Longitude.max()]
			request.session['sw'], request.session['ne'] = sw, ne 
			request.session['endindex'] = len(trackDataFrames[int(request.session['selectedTrack'])])-1
			return HttpResponseRedirect(reverse('home'))
		else:
			request.session['endindex'] = len(trackDataFrames[int(request.session['selectedTrack'])])-1
			return HttpResponseRedirect(reverse('home'))

		
def mainpage(request):
	global default_ne, default_sw
	#request.session = {}
	
	#print('*****************', request.session['selectedTrack'])
	
	cf = forms.FileListForm(choices=[])
	if 'listofchoices' not in request.session:
		request.session['listofchoices'] = json.dumps([])
	if 'trackDataFrames' not in request.session:
		request.session['trackDataFrames'] = []
	if 'ne' not in request.session:
		request.session['ne'] = default_ne
	if 'sw' not in request.session:
		request.session['sw'] = default_sw
		
	ne, sw = request.session['ne'], request.session['sw']
	
	if len(ne) == 0 or len(sw) == 0:
		ne, sw = default_ne, default_sw
	
	listofchoices = json.loads(request.session['listofchoices'])
	
	if request.method == 'POST':
		selectFileForm = forms.SelectTrackForm(request.POST, request.FILES)
		if selectFileForm.is_valid():
			gpxfilename = request.FILES['docfile']
			listofchoices.append((str(len(listofchoices)), gpxfilename.name))
			request.session['listofchoices'] = json.dumps(listofchoices)
			trackDataFrame = pdframe.GetDataFrame( gpxfilename, morecols=True )
			y = trackDataFrame.VarLoad.ewm(span=10).mean()
			y[0] = y[1]
			y.iloc[-1] = y.iloc[-2]
			trackDataFrame['VarLoadSmooth'] = y
			request.session['trackDataFrames'].append(trackDataFrame.to_json())
			request.session['selectedTrack'] = len(request.session['trackDataFrames'])-1;
			request.session['endindex'] = trackDataFrame.shape[0]-1
			trackDataFrames = [ pd.read_json(x) for x in request.session['trackDataFrames'] ]
						
			sw = [ min([trackDataFrames[i].Latitude.min() for i in range(len(trackDataFrames))]),\
				min([trackDataFrames[i].Longitude.min() for i in range(len(trackDataFrames))]) ]
			ne = [ max([trackDataFrames[i].Latitude.max() for i in range(len(trackDataFrames))]),\
				max([trackDataFrames[i].Longitude.max() for i in range(len(trackDataFrames))]) ]
			request.session['sw'], request.session['ne'] = sw, ne 
			# recompute the colors
			return HttpResponseRedirect(reverse(mainpage))
	else:
		selectFileForm = forms.SelectTrackForm()
		cf = forms.FileListForm(choices=listofchoices)

		

	trackDataFrames = [ pd.read_json(x) for x in request.session['trackDataFrames'] ]

	#~ if 'trackDataFrames' in request.session and len(request.session['trackDataFrames']) > 0:
		#~ sw = [ min([trackDataFrames[i].Latitude.min() for i in range(len(trackDataFrames))]),\
			#~ min([trackDataFrames[i].Longitude.min() for i in range(len(trackDataFrames))]) ]
		#~ ne = [ max([trackDataFrames[i].Latitude.max() for i in range(len(trackDataFrames))]),\
			#~ max([trackDataFrames[i].Longitude.max() for i in range(len(trackDataFrames))]) ]
		#~ request.session['sw'], request.session['sw'] = sw, ne 

	if 'selectedTrack' not in request.session:
		request.session['selectedTrack'] = -1
	if 'endindex' not in request.session:
		request.session['endindex'] = -1


	trackDF4Web, colorsBounds = DefinesTracks4Web(request)
		
	colors=[]
	for col in colorsBounds:
		colors.append((colorsBounds[col], col))
	colors.sort()
	colorsName = [ x for _,x in colors ]
	colorsBound = [ x for x,_ in colors ]


	#print('*****************', request.session['selectedTrack'])
	
	if(int(request.session['selectedTrack']) >= 0):
		cf.fields['filelist'].initial = request.session['selectedTrack']
		
	

	return render(request, 'main.html', {'sw': sw, 'ne': ne, 'selfileform': selectFileForm,\
		'dataframes': json.dumps(trackDF4Web),\
		'filelist': cf, 'selected': int(request.session['selectedTrack']), 'endindex': request.session['endindex'],
		'colorsname': colorsName, 'colorsbound': colorsBound})
		

