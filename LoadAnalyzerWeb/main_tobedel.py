# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:40:19 2016

@author: gianluca
"""



import pandas as pd
import numpy

import osmtiles
import base64
import math

try:
    from geopy.distance import great_circle
except ImportError:
    from geopy.distance import GreatCircleDistance as great_circle

try:
    # Python2
    import Tkinter as tk
    import tkFileDialog as filedialog
    from urllib2 import urlopen
except ImportError:
    # Python3
    import tkinter as tk
    from tkinter import filedialog
    from urllib.request import urlopen
    
import pdframe
import os
import bisect

import threading





# map canvas size
canvasWidth, canvasHeigth = 800, 600
# map zoom
zoom = None
# It is used to load the first tile at the boot
# or is the nw point of the bounding box containing
# all the tracks
nwLon, nwLat, seLon, seLat = -6.0, 56.0, 19, 35
# The starting tiles (NW)
nwTileX, nwTileY =  None, None
numXTiles, numYTiles = None, None
# the position on the screen corresponding to nwLon, nwLat
# this will became the northwest point 
nwX, nwY = None, None

# The bounding box containing the traks will be shifted
# on left and down by these values
shiftX, shiftY = 0, 0

# id of the loaded tiles
tilesId = []
# id of lines describing tracks (canvasId, trackId)
segsId = []

# Position on the screen of nwTile
nwTilePoSX, nwTilePosY = None, None

# x and y scrolling size in pixels
scrollX, scrollY = 10, 10

# the size of tiles
tileSize=256

# the tiles dictionary save le loaded tiles in the following format
#
# tiles[(z,i,j)] = image
#
# where (z,i,j) are the three integer in the tile url 
tiles={}

#List of all dataframes coming from the loaded gpx tracks
trackDataFrames =[]

# opencyclemap max zoom
maxZoom = 20

# Rectangle selection vertices
selStartX, selStartY, selFinishX, selFinishY = None, None, None, None
idSelRect = None
selBBox = None

# the index of the selected track
selectedTrackIndex = None
# size of displayed segments in canvas
normalSegSize, selectedSegSize = 2, 5

# widget containing the coordinates of mouse pointer
lonlatText, bglonlatText = None, None

######################################
# Selectiong of a fragment  of a track,
####
# coordinate (in the canvas) of the two poins
selectionStartXY, selectionEndXY = None, None
#id of the graphic elements
selectionStartCanvas, selectionEndCanvas = None, None
# indices inside the track dataframe
selectionStartIndex, selectionEndIndex = None, None
# scale widgets
scrollSelectionStart, scrollSelectionEnd = None, None
#
selStartIcon, selEndIcon = None, None
#
iconSize = 7


# String appearing inthe textarea
txtMessage = None

# the start and end line  on the plotcanvas conresponding the
# the selecction on track
plotcanvasStartline, plotcanvasEndline = None, None
colorSpeed, colorLoad = 'brown', 'cyan'

#this is the list of id widget loaded into the canvas, are removed
#when the map is redisplayed
# todo...


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



# delete the selected track (selectedTrackIndex)
def DeleteTrack():
    global trackDataFrames, selectedTrackIndex, segsId
    
    idTrackToBeDel = id(trackDataFrames[selectedTrackIndex])
    trackDataFrames.pop(selectedTrackIndex)
    listTrack.delete(selectedTrackIndex)
    
    # delete track from canvas and from segsId list
    for x in range(len(segsId))[-1::-1]:
         idCanvas, idTrack = segsId[x]
         if idTrack == idTrackToBeDel:
             canvas.delete(idCanvas)
             del(segsId[x])
        
    DisplayTracks()


def DisplayTracks(attribute = 'VarLoadSmooth'):
    global nwLat, nwLon, nwTileX, nwTileY, nwX, nwY
    global numXTiles, numYTiles
    global normalSegSize, selectedSegSize, segsId
    global shiftX, shiftY
    
    if len(trackDataFrames) > 0:        
        mins = [ trackDataFrames[i].VarLoadSmooth.min() for i in range(len(trackDataFrames))]
        maxs = [ trackDataFrames[i].VarLoadSmooth.max() for i in range(len(trackDataFrames))]    
        
        #segments = [i*1.0*(max(maxs)-min(mins))/4 for i in [0, 1.0, 2.0, 3.0, 4.0] ]    
        tmin, tmax = min(mins), max(maxs)        
        segments = numpy.arange(tmin, tmax, (tmax-tmin)/5)
        colors = ['yellow', 'cyan', 'lightgreen', 'red', 'black']    
        
        trackDFindex = 0
        for trackDF in trackDataFrames:
            for i in trackDF.index[1:]:
                    p0xy = LonLat2PositionInCanvas(trackDF.ix[i-1].Longitude,\
                        trackDF.ix[i-1].Latitude)
                    p1xy = LonLat2PositionInCanvas(trackDF.ix[i].Longitude,\
                        trackDF.ix[i].Latitude)
                    #try:
                    color = colors[bisect.bisect_right(segments, trackDF.ix[i][attribute])-1]
                    #except IndexError:
                    #    color = colors[-1]
                    segsId.append((tk.Canvas.create_line(canvas, p0xy[0], p0xy[1],\
                        p1xy[0], p1xy[1], fill=color, width=normalSegSize,\
                        tags = ['track'+str(id(trackDF)),\
                        'linetracks',\
                        str(trackDF.ix[i-1].Longitude),\
                        str(trackDF.ix[i-1].Latitude),\
                        str(trackDF.ix[i].Longitude),\
                        str(trackDF.ix[i].Latitude)]),\
                        id(trackDF)))
            trackDFindex += 1
        #canvas.move('linetracks', x, y)
        WidgetMove('linetracks', shiftX, shiftY)
        #ScrollLeft( scroll = x )
        #ScrollUp( scroll = y )

# Set the selectionEndIndex according to the Scale-widget status
def CommandScaleSelEnd(howmuch):
    global selectionStartIndex, selectionEndIndex
    global selectedTrackIndex
    global selectionEndCanvas
    global shiftX, shiftY
    global plotcanvas, plotcanvasStartline, plotcanvasEndline   
    
    try:
        selectionEndIndex = int(howmuch)
        if(selectionStartIndex < selectionEndIndex):
            SetTextMessage()
            
            selectionEndXY = LonLat2PositionInCanvas(trackDataFrames[selectedTrackIndex].Longitude.iloc[selectionEndIndex],\
                trackDataFrames[selectedTrackIndex].Latitude.iloc[selectionEndIndex])
            
            canvas.coords(selectionEndCanvas,\
                selectionEndXY[0]-iconSize+shiftX,
                selectionEndXY[1]-iconSize+shiftY,\
                selectionEndXY[0]+iconSize+shiftX,
                selectionEndXY[1]+iconSize+shiftY,\
                )
            
            taglist  = plotcanvas.gettags(plotcanvasEndline)
            w, h, left, right, up, down = \
                float(taglist[0]), float(taglist[1]),\
                float(taglist[2]), float(taglist[3]),\
                float(taglist[4]), float(taglist[5])
                        
            track = trackDataFrames[selectedTrackIndex] 
            
            plotcanvas.coords(plotcanvasEndline,\
                left+w*track.TotDistance[selectionEndIndex]/track.TotDistance.max(), up,\
                left+w*track.TotDistance[selectionEndIndex]/track.TotDistance.max(), down)

    except TypeError:
        ''
    
        
# Set the selectionStartIndex according to the Scale-widget status
def CommandScaleSelStart(howmuch):
    global selectionStartIndex, selectionEndIndex
    global selectedTrackIndex
    global selectionStartCanvas
    global plotcanvas, plotcanvasStartline, plotcanvasEndline    
    
    try:
        selectionStartIndex = int(howmuch)
        if(selectionStartIndex < selectionEndIndex):
            SetTextMessage()
            
            selectionStartXY = LonLat2PositionInCanvas(trackDataFrames[selectedTrackIndex].Longitude.iloc[selectionStartIndex],\
                trackDataFrames[selectedTrackIndex].Latitude.iloc[selectionStartIndex])
            
            canvas.coords(selectionStartCanvas,\
                selectionStartXY[0]-iconSize+shiftX,\
                selectionStartXY[1]-iconSize+shiftY,\
                selectionStartXY[0]+iconSize+shiftX,\
                selectionStartXY[1]+iconSize+shiftY,\
                )  
            taglist  = plotcanvas.gettags(plotcanvasStartline)
            w, h, left, right, up, down = \
                float(taglist[0]), float(taglist[1]),\
                float(taglist[2]), float(taglist[3]),\
                float(taglist[4]), float(taglist[5])
                        
            track = trackDataFrames[selectedTrackIndex] 
            
            plotcanvas.coords(plotcanvasStartline,\
                left+w*track.TotDistance[selectionStartIndex]/track.TotDistance.max(), up,\
                left+w*track.TotDistance[selectionStartIndex]/track.TotDistance.max(), down)
    except TypeError:
        ''  
   
def AddTile(i, j):
    global nwLat, nwLon, nwTileX, nwTileY, nwX, nwY, zoom
    global numXTiles, numYTiles
    global nwTilePoSX, nwTilePosY
    global shiftX, shiftY        
        
    try:
        try:
            tile = tiles[(zoom,nwTileX+i,nwTileY+j)]
        except KeyError:
            pngurl = osmtiles.tileURL(nwTileX+i,nwTileY+j,zoom,'cyclemap')
            image_byt = urlopen(pngurl).read()
            try:
                image_b64 = base64.encodebytes(image_byt)
            except AttributeError:
                image_b64 = base64.encodestring(image_byt)
            tile = tk.PhotoImage(data=image_b64, format='png')
            tiles[(zoom,nwTileX+i,nwTileY+j)] = tile
        
        tileid = tk.Canvas.create_image(canvas, tileSize*i+nwX,\
            tileSize*j+nwY, image = tile, anchor='nw',tag =  'tiles')
        #tilesId.append(tileid)
        #canvas.tag_lower(tileid)
    except Exception as excpt:
        print('******'+type(excpt).__name__)
        if type(excpt).__name__ == 'URLError':
            tileid = tk.Canvas.create_rectangle(canvas,\
                tileSize*i+nwX, tileSize*j+nwY,\
                tileSize*i+nwX+tileSize, tileSize*j+nwY+tileSize,
                fill='', outline='red', tag = 'tiles')  
            
    tilesId.append(tileid)
    canvas.tag_lower(tileid)
    #canvas.move(tileid, x, y)
    WidgetMove(tileid, shiftX, shiftY)
   
def LoadTiles(deleteall = False ):
    global nwLat, nwLon, nwTileX, nwTileY, nwX, nwY, zoom
    global numXTiles, numYTiles
    global nwTilePoSX, nwTilePosY
    global lonlatText, bglonlatText
    
        
    if deleteall:
        canvas.delete('all')
    else:
        canvas.delete('tiles')
        canvas.delete('markers')

    for i in range(-1,int(numXTiles)+1):
        for j in range(-1,int(numYTiles)+1):
            #AddTile( i, j, x, y )
            t = threading.Thread(target=AddTile, args = (i, j))
            t.deamon = True
            t.start()
                                        
    lonlatText = canvas.create_text(10, 10,anchor='nw', text='')
    bglonlatText = canvas.create_rectangle( canvas.bbox(lonlatText),\
        fill='white', outline='white' )
    canvas.tag_lower(bglonlatText,lonlatText)


# Shows information of selected track     
def ListTrackOnSelect(event):
    global canvas, selectedTrackIndex
    global selectionStartIndex, selectionEndIndex
    global scrollSelectionStart, scrollSelectionEnd
    
    if len(trackDataFrames) > 0:
        w = event.widget
        selectedTrackIndex = int(w.curselection()[0])
        #value = w.get(selectedTrackIndex)
    
        seltrack = trackDataFrames[selectedTrackIndex]
    
        
        buttonDeleteTrack.configure(state=tk.NORMAL)
        buttonZoomOnTrack.configure(state=tk.NORMAL)
        scrollSelectionStart.configure(state=tk.NORMAL)
        scrollSelectionEnd.configure(state=tk.NORMAL)
        
        selectionStartIndex, selectionEndIndex = 0, seltrack.shape[0]-1
        scrollSelectionStart.configure(state=tk.NORMAL, from_ = selectionStartIndex,\
            to = selectionEndIndex)
        scrollSelectionEnd.configure(state=tk.NORMAL, from_ = selectionStartIndex,\
            to = selectionEndIndex)
        
        scrollSelectionEnd.set(selectionEndIndex)
        scrollSelectionStart.set(0)
        
        SetTextMessage()
        
        ReDisplayTracks(startindex = 0, endindex = selectionEndIndex)
        
        PlotLoadSpeed(seltrack)

def LoadGpxFile():
    global canvas, nwX, nwY
    
    dlg = filedialog.Open(root, filetypes=[('Gpx file', '*.gpx')])
    gpxfilename = dlg.show()

    if gpxfilename != '':
        SetTextMessage('Reading file...')
        trackDataFrame = pdframe.GetDataFrame( gpxfilename, morecols=True )
        SetTextMessage('Smooting...')
#        if trackDataFrame.shape[0] >= 20:
#            #y = pdframe.savitzky_golay(trackDataFrame.VarLoad.tolist(), 9, 2)
#            #y = pd.stats.moments.ewma(trackDataFrames.VarLoad)
#            y = trackDataFrame.VarLoad.ewm(span=10).mean()
#        else:
#            y = trackDataFrame.VarLoad.tolist()
        y = trackDataFrame.VarLoad.ewm(span=10).mean()
        # to remove border effects        
        y[0] = y[1]
        y.iloc[-1] = y.iloc[-2]
        trackDataFrame['VarLoadSmooth'] = y
        trackDataFrames.append(trackDataFrame)
        # Defines
        #      nwLon, nwLat, zoom, nwTileX, nwTileY
        # according to the tracks.
        #       w e h are the size, in pixels of the bounding box
        # are used to avoid that the tracks are placed in
        # the upper-left corner of the canvas
        SetViewToBoundingBox()
        SetTextMessage('Loading tiles...')       
        LoadTiles( deleteall = True )
        SetTextMessage('Painting tracks...')
        DisplayTracks( )
        #nwX += x
        #nwY += y
        listTrack.insert(tk.END, os.path.basename(gpxfilename))
        SetTextMessage('Done!')
        
def LonLat2PositionInCanvas(lon, lat):
    global nwLat, nwLon, nwTileX, nwTileY, nwX, nwY
    global numXTiles, numYTiles, zoom
    # tile containing p=(lon,lat)
    tx, ty = osmtiles.tileXY(lat, lon, zoom)
    # the position of corners of t    
    south, west, north, east = osmtiles.tileEdges(tx,ty,zoom)
    # the sizes of t in meters (passing in lon,lat)  
    msize_t_x, msize_t_y = great_circle((lat,west), (lat,east)).meters,\
        great_circle((north,lon), (south,lon)).meters
    # distance from p to NW of tile t in meters
    mdist_pt_x, mdist_pt_y = great_circle((lat,west), (lat,lon)).meters,\
        great_circle((north,lon), (lat,lon)).meters
    # dx, dy is the distance in pixels from (west,north)
    # dx:tileSize = mdist_pt_x:msize_t_x
    dx, dy = 1.0*tileSize*mdist_pt_x/msize_t_x,\
        1.0*tileSize*mdist_pt_y/msize_t_y
    # x, y are the posizion in cvs  
    px, py = nwX + tileSize*(tx-nwTileX) + dx,\
        nwY + tileSize*(ty-nwTileY) + dy

    return px, py

# Scroll the canvas in the way that the rectangle
# of size w x h in the upper-right corner will be on the center
def MoveToCenter(w, h):
    x,y = (canvas.winfo_width() - w)/2, (canvas.winfo_height()-h)/2

    ScrollLeft(scroll=x)
    ScrollUp(scroll=y)    

def PlotLoadSpeed(track):
    global plotcanvas
    global plotcanvasStartline, plotcanvasEndline
    
    plotcanvas.delete('all')
    
    w, h =  plotcanvas.winfo_width(), plotcanvas.winfo_height()
    cor = max(w,h)/30
    w, h = w-4*cor, h-3*cor
    
    left, right, up, down = 2*cor, 2*cor+w, cor, cor+h     
    
    plotcanvas.create_line(left, up, right, up, fill = 'black')
    plotcanvas.create_line(right, up, right, down, fill = 'black')
    plotcanvas.create_line(right, down, left, down, fill = 'black')
    plotcanvas.create_line(left, down, left, up, fill = 'black')
     
    plotcanvasStartline= plotcanvas.create_line(\
        left+w*track.TotDistance[0]/track.TotDistance.max(), up,\
        left+w*track.TotDistance[0]/track.TotDistance.max(), down,\
        fill='blue', tags = [ str(w), str(h), str(left),str(right), str(up), str(down)])

    plotcanvasEndline= plotcanvas.create_line(\
        left+w*track.TotDistance[track.shape[0]-1]/track.TotDistance.max(), up,\
        left+w*track.TotDistance[track.shape[0]-1]/track.TotDistance.max(), down,\
        fill='red', tags = [ str(w), str(h), str(left),str(right), str(up), str(down)])
    
    hticknum, ticksize = 5, 5
    
    dist = track.TotDistance    
    load = track.VarLoadSmooth 
    speed = track.Speed.ewm(span=10).mean()
            
    maxdist = dist.iloc[-1]
    
    # horizontal ticks    
    for i in range(hticknum):
        plotcanvas.create_line(left+(w/hticknum)*i, up,\
            left+(w/hticknum)*i, up+ticksize, fill='black')
        plotcanvas.create_line(left+(w/hticknum)*i, down,\
            left+(w/hticknum)*i, down-ticksize, fill='black')        

    vticknum = 5
           
    
    #vertical ticks
    for i in range(vticknum):
        plotcanvas.create_line( left, down-(h/vticknum)*i,\
            left+ticksize, down-(h/vticknum)*i, fill = 'black')
        plotcanvas.create_text(left, down-(h/vticknum)*i,\
            text=str(int(i*load.max()/vticknum)), anchor = tk.E,\
            fill = colorLoad)
        plotcanvas.create_line( right, down-(h/vticknum)*i,\
            right-ticksize, down-(h/vticknum)*i, fill = 'black')   
        plotcanvas.create_text(right, down-(h/vticknum)*i,\
            text=str(int(i*speed.max()/vticknum)), anchor = tk.W,\
            fill = colorSpeed)

    # scalingon y to avoid tersect with the up border

    sc = 0.97
    for i in range(0,track.shape[0]-1):
        plotcanvas.create_line(\
            left+w*track.TotDistance[i]/maxdist,\
            down-sc*h*load[i]/load.max(),\
            left+w*track.TotDistance[i+1]/maxdist,\
            down-sc*h*load[i+1]/load.max(), fill = colorLoad, width=1.5)
        plotcanvas.create_line(\
            left+w*track.TotDistance[i]/maxdist,\
            down-sc*h*speed[i]/speed.max(),\
            left+w*track.TotDistance[i+1]/maxdist,\
            down-sc*h*speed[i+1]/speed.max(), fill = colorSpeed, width=1.3)

    plotcanvas.create_text((w+3*cor)/2, h+(cor), anchor=tk.N,\
        justify=tk.CENTER, text='Distance vs Speed (brown) and Load (cyan)')

def PosInCanvas2LatLon(x, y):
    global nwLat, nwLon, nwTileX, nwTileY, nwX, nwY
    x -= shiftX
    y -= shiftY
    # tile containing (x,y)
    tX, tY = math.floor((1.0*x-nwX)/tileSize)+nwTileX,\
        math.floor((1.0*y-nwY)/tileSize)+nwTileY
    # position in the canvas of the nw angle of the tile
    tPX, tPY = math.floor((x-nwX)/tileSize)*tileSize+nwX,\
        math.floor((y-nwY)/tileSize)*tileSize+nwY

    marcX, marcY = tX + (1.0*x-tPX)/tileSize, tY + (1.0*y-tPY)/tileSize

    return osmtiles.xy2latlon(marcX, marcY, zoom)

# redisplay tracks taking into account the selection of track
def ReDisplayTracks(startindex = 0, endindex = -1):
    global normalSegSize, selectedSegSize, selectedTrackIndex, trackDataFrames
    global selectionStartXY, selectionEndXY, selectedSegSize
    global selectionStartIndex, selectionEndIndex
    global selectionStartCanvas, selectionEndCanvas
    global selStartIcon, selEndIcon
    global shiftX, shiftY
    
    if selectedTrackIndex != None:
        canvas.itemconfig('linetracks', width=normalSegSize)
        x = 'track'+str(id(trackDataFrames[selectedTrackIndex]))        
        canvas.itemconfig(x, width=selectedSegSize)
        canvas.tag_raise(x)
                        
        # Position of special signs to thestart and end of track
        selectionStartIndex, selectionEndIndex = startindex, endindex                
                    
        selectionStartXY = LonLat2PositionInCanvas(trackDataFrames[selectedTrackIndex].Longitude.iloc[selectionStartIndex],\
            trackDataFrames[selectedTrackIndex].Latitude.iloc[selectionStartIndex])
        selectionEndXY = LonLat2PositionInCanvas(trackDataFrames[selectedTrackIndex].Longitude.iloc[selectionEndIndex],\
            trackDataFrames[selectedTrackIndex].Latitude.iloc[selectionEndIndex])
            
        canvas.delete(selectionStartCanvas)
        canvas.delete(selectionEndCanvas)
         
        selectionStartCanvas = tk.Canvas.create_oval(canvas, selectionStartXY[0]-iconSize,\
                        selectionStartXY[1]-iconSize, selectionStartXY[0]+iconSize,\
                        selectionStartXY[1]+iconSize, tag = 'markers', fill='',\
                        outline='blue', width=2)        
            
        selectionEndCanvas = tk.Canvas.create_rectangle(canvas, selectionEndXY[0]-iconSize,\
                        selectionEndXY[1]-iconSize, selectionEndXY[0]+iconSize,\
                        selectionEndXY[1]+iconSize, tag = 'markers', fill='',
                        outline='red', width=2)  
        
        WidgetMove('markers', shiftX, shiftY)
    
def ResizeCanvas(event):
    geomtxt = root.wm_geometry()
    w = int(geomtxt.split('x')[0])
    h = int(geomtxt.split('x')[1].split('+')[0])
    canvas.configure(width = w, height = h)
    #todo    
    #DisplayMap()
    #UpdateTracks()
   
def ScrollDown(event = None, scroll=scrollY):
    global nwX
    global nwLon, nwLat, seLon, seLat
    global selectionStartCanvas, selectionEndCanvas   
    
    nwLat = nwLat + (seLat -nwLat)/10
    seLat = seLat + (seLat -nwLat)/10
    SetViewToBoundingBox(west = nwLon, north = nwLat, east = seLon, south = seLat)
    LoadTiles(deleteall = False)
    ZoomTracks()
    ReDisplayTracks()

def ScrollLeft(event = None, scroll=scrollX):
    global nwX
    global nwLon, nwLat, seLon, seLat
    global selectionStartCanvas, selectionEndCanvas   
    
    nwLon = nwLon - (seLon -nwLon)/10
    seLon = seLon - (seLon -nwLon)/10
    SetViewToBoundingBox(west = nwLon, north = nwLat, east = seLon, south = seLat)
    LoadTiles(deleteall = False)
    ZoomTracks()
    ReDisplayTracks()
      
def ScrollRight(event = None, scroll=scrollX):
    global nwX
    global nwLon, nwLat, seLon, seLat
    global selectionStartCanvas, selectionEndCanvas   
    
    nwLon = nwLon + (seLon -nwLon)/10
    seLon = seLon + (seLon -nwLon)/10
    SetViewToBoundingBox(west = nwLon, north = nwLat, east = seLon, south = seLat)
    LoadTiles(deleteall = False)
    ZoomTracks()
    ReDisplayTracks()

def ScrollUp(event = None, scroll = scrollY):
    global nwX
    global nwLon, nwLat, seLon, seLat
    global selectionStartCanvas, selectionEndCanvas   
    
    nwLat = nwLat - (seLat -nwLat)/10
    seLat = seLat - (seLat -nwLat)/10
    SetViewToBoundingBox(west = nwLon, north = nwLat, east = seLon, south = seLat)
    LoadTiles(deleteall = False)
    ZoomTracks()
    ReDisplayTracks()

def SelectionFinish(event):
    global idSelRect, selStartX, selStartY, selFinishX, selFinishY, selBBox,zoom
    
    selFinishX, selFinishY = event.x, event.y    
    
    sel0lat, sel0lon = PosInCanvas2LatLon(selStartX, selStartY)
    sel1lat, sel1lon = PosInCanvas2LatLon(selFinishX, selFinishY)
    
    selBBox = min(sel0lon, sel1lon), max(sel0lat, sel1lat),\
        max(sel0lon, sel1lon), min(sel0lat, sel1lat)
        
    SetViewToBoundingBox(west=selBBox[0], north=selBBox[1],
                         east=selBBox[2], south=selBBox[3])
    
    LoadTiles(deleteall = False)
    ZoomTracks()
    
    tk.Canvas.delete(canvas, idSelRect)
    idSelRect, selStartX, selStartY = None, None, None

def SelectionStart(event):
    global selStartX, selStartY, selFinishX, selFinishY
    
    selStartX, selStartY = event.x, event.y
    
def SelectionUpdate(event):
    global idSelRect
    global selStartX, selStartY
    global lonlatText, bglonlatText    
    
    mlat, mlon =  PosInCanvas2LatLon(event.x, event.y)    
    lonlatstring = "{:.5f}".format(mlon)+","+"{:.5f}".format(mlat)
    canvas.itemconfig(lonlatText, text=lonlatstring)
    canvas.delete(bglonlatText)
    bglonlatText = canvas.create_rectangle(canvas.bbox(lonlatText),fill='white', outline='white')
    canvas.tag_lower(bglonlatText,lonlatText)

    
    if selStartX != None:
        x,y = event.x, event.y
        if idSelRect != None:
            tk.Canvas.delete(canvas, idSelRect)
        
        idSelRect = tk.Canvas.create_rectangle(canvas, min(selStartX, x),\
            min(selStartY, y), max(selStartX, x), max(selStartY, y))   
   
def SetFocus(event):
    event.widget.focus_set()
    
# set the message related to the selection
def SetTextMessage(msg = ''):
    global selectionStartIndex, selectionEndIndex, selectedTrackIndex
    global txtMessage

    if msg != '':
        messageString = msg
        color = 'red'
    else:    
        selTrackDF = trackDataFrames[selectedTrackIndex]
        
        trackMeters = selTrackDF.TotDistance.iloc[selectionEndIndex] -\
            selTrackDF.TotDistance.iloc[selectionStartIndex] 
        trackSecs = selTrackDF.Seconds.iloc[selectionEndIndex] -\
            selTrackDF.Seconds.iloc[selectionStartIndex]
        trackSpeed = 3.6*trackMeters/trackSecs
        
        totLoad = selTrackDF.VarLoad[selectionStartIndex:selectionEndIndex].sum()
        
        messageString = "Length: %d m." % trackMeters
        messageString += '\n'
        messageString += "Time: %2dmin%2dsec" % ( int(trackSecs)/60, int(trackSecs) - int((trackSecs)/60)*60 )
        messageString += '\n'
        messageString += "Speed: %.1f km/h" % trackSpeed
        messageString += '\n'
        messageString += "Load: %f" % totLoad
        messageString += '\n'
        messageString += "Max. load: %d" %  selTrackDF.VarLoad[selectionStartIndex:selectionEndIndex].max()
        messageString += '\n'
        messageString += "Avr. load: %f" % (float(totLoad)/(selectionEndIndex-selectionStartIndex))
        color = 'black'        
        
    textarea.config(fg=color)
    txtMessage.set(messageString)
    root.update()
    

               
def SetViewToBoundingBox(west = None, north = None, east = None, south = None):
    global nwLat, nwLon, seLat, seLon, zoom, nwTileX, nwTileY
    global canvasWidth, canvasHeigth,tileSize
    global nwX, nwY, numXTiles, numYTiles
    global shiftX, shiftY

    if west == None:    
        west, north, east, south = 1000, -1000, -1000, 1000
    
        for df in trackDataFrames:
            west, north, east, south = \
                min(west, df.Longitude.min()), max(north, df.Latitude.max()),\
                max(east, df.Longitude.max()), min(south, df.Latitude.min()),\
            
    nwLon, nwLat, seLon, seLat = west, north, east, south
    # computing the new zoom
    findIt = False
    zoom = maxZoom
    while zoom > 0 and not findIt:    
        nwTX, nwTY =  osmtiles.latlon2xy(nwLat, nwLon, zoom)
        seTX, seTY =  osmtiles.latlon2xy(south, east, zoom)
        # size in pixels to dispaly the track with the current zoom
        sizex, sizey =  (seTX - nwTX)*tileSize, (seTY - nwTY)*tileSize
        
        if sizex < canvas.winfo_width() and sizey < canvas.winfo_height():
            findIt = True
        else:
            zoom -= 1  
    
    px, py = osmtiles.latlon2relativePixels(nwLat, nwLon, zoom)
    nwX, nwY = -px, -py
    
    nwTileX, nwTileY = osmtiles.tileXY(nwLat, nwLon, zoom)
    numXTiles, numYTiles = math.ceil(float(sizex)/tileSize)+2,\
        math.ceil(float(sizey)/tileSize)+2

    shiftX, shiftY = (canvas.winfo_width() - sizex)/2,\
        (canvas.winfo_height()-sizey)/2        


# Zoom on the selectedtrack
def ZoomOnTrack():
    global trackDataFrames, selectedTrackIndex, segsId
    global selectionStartIndex, selectionEndIndex    
        
    track = trackDataFrames[selectedTrackIndex]
    # zoomin on track
    SetTextMessage('Computing new bounding box...')
    SetViewToBoundingBox(
        west = track.Longitude[selectionStartIndex:selectionEndIndex].min(),\
        north = track.Latitude[selectionStartIndex:selectionEndIndex].max(),\
        east = track.Longitude[selectionStartIndex:selectionEndIndex].max(),\
        south = track.Latitude[selectionStartIndex:selectionEndIndex].min())

    SetTextMessage('Loading tiles...')
    LoadTiles(deleteall = False)
     
    SetTextMessage('Zooming tracks...')    
    #DisplayTracks()

    ZoomTracks()    
    SetTextMessage('Highlighting track...')
    ReDisplayTracks(startindex = selectionStartIndex,\
        endindex = selectionEndIndex)
    SetTextMessage('Done!')

# Zoom on the tracks
def ZoomOnTracks():
    global trackDataFrames, selectedTrackIndex, segsId

    SetViewToBoundingBox()
    LoadTiles(deleteall = False)
    #MoveToCenter(w,h)
    #DisplayTracks()
    ZoomTracks()

def ZoomTracks():
    global shiftX, shiftY
    x = canvas.find_withtag('linetracks')
    for  l in x:
        idtags = canvas.gettags(l)            
        lon0, lat0, lon1, lat1 = idtags[2:]
        x0, y0 = LonLat2PositionInCanvas(float(lon0), float(lat0))
        x1, y1 = LonLat2PositionInCanvas(float(lon1), float(lat1)) 
        canvas.coords(l, x0+shiftX, y0+shiftY,\
            x1+shiftX, y1+shiftY)

def WidgetMove(idOrTag, x, y):
    global nwX, nwY , canvas
    canvas.move(idOrTag, x, y)
    #nwX += x
    #nwY += y

#    if selectedTrackIndex != None:
#        for canvasId, _ in segsId:
#            idtags = canvas.gettags(canvasId)
#            #print(idtags)
#            lon0, lat0, lon1, lat1 = idtags[2:]
#            x0, y0 = LonLat2PositionInCanvas(float(lon0), float(lat0))
#            x1, y1 = LonLat2PositionInCanvas(float(lon1), float(lat1)) 
#            canvas.coords(canvasId, x0, y0, x1, y1)

####################################################
############################################### MAIN
          
root = tk.Tk()
root.wm_title('LoadAnalyzer')
#
root.geometry("1000x700+0+0")
#pad=3
#root.geometry("{0}x{1}+0+0".format(\
#            root.winfo_screenwidth()-pad,\
#            root.winfo_screenheight()-pad))
root.columnconfigure(1, weight=1)
root.rowconfigure(1, weight=1)
root.columnconfigure(7, minsize=100)
root.columnconfigure(8, minsize=100)
root.columnconfigure(9, minsize=100)

canvas = tk.Canvas(root, borderwidth=1, \
            highlightthickness=1, bg="white")

canvas.grid(column=0, columnspan=6, rowspan=8, row=0)



#selStartIcon = tk.PhotoImage(file='startsel.png', format='png')
#selEndIcon = tk.PhotoImage(file='endsel.png', format='png')


canvas.bind('<Up>', ScrollUp)
canvas.bind('<Down>', ScrollDown)
canvas.bind('<Left>', ScrollLeft)
canvas.bind('<Right>', ScrollRight)
canvas.bind('<Button-3>', SelectionStart)
canvas.bind('<Motion>', SelectionUpdate)
canvas.bind('<ButtonRelease-3>', SelectionFinish)

root.bind('<Button-1>', SetFocus)

#
tk.Message(root, text='Tracks', width=200).grid(column=7, row=0,\
    rowspan=1, columnspan=3)



listTrack = tk.Listbox(root, width=40)
listTrack.grid(column=7, row=1, rowspan=1, sticky=tk.N, columnspan=3)
listTrack.bind('<<ListboxSelect>>', ListTrackOnSelect) 
#
buttonDeleteTrack = tk.Button(root, text="Delete", command=DeleteTrack,\
    state=tk.DISABLED, justify=tk.LEFT)
buttonDeleteTrack.grid(column=7, row=2)
#
buttonZoomOnTrack = tk.Button(root, text="Zoom", command=ZoomOnTrack,\
    state=tk.DISABLED,  justify=tk.LEFT)
buttonZoomOnTrack.grid(column=8, row=2)
#
buttonZoomOut = tk.Button(root, text="Zoom out", command=ZoomOnTracks,\
    state=tk.NORMAL,  justify=tk.LEFT)
buttonZoomOut.grid(column=9, row=2)


#
#scrollbar for start selection
scrollSelectionStart = tk.Scale(root, orient=tk.HORIZONTAL,\
    command=CommandScaleSelStart, label="Start",\
    state=tk.DISABLED, showvalue=0)
scrollSelectionStart.grid(column=7, row=3, rowspan=1, columnspan=1)
scrollSelectionEnd = tk.Scale(root, orient=tk.HORIZONTAL,\
    command=CommandScaleSelEnd, label="End",\
    state=tk.DISABLED, showvalue=0)
scrollSelectionEnd.grid(column=9, row=3, rowspan=1, columnspan=1)


######### plot distance vs load and speed
plotcanvas = tk.Canvas(root, borderwidth=1, \
            highlightthickness=1, bg="white", width=280, height=220)

plotcanvas.grid(column=7, columnspan=3, rowspan=2, row=4)



#f = Figure(figsize=(3,2), dpi=100)
#a = f.add_subplot(111)
#a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])
#plot = FigureCanvasTkAgg(f, master=root)
#plot.get_tk_widget().grid(column=7, row=4, rowspan=2, columnspan=3)
#plot.show()


#
txtMessage = tk.StringVar()
txtMessage.set('')
textarea=tk.Label(root, textvariable = txtMessage, width=20, height=20, justify=tk.LEFT)
textarea.grid(column=7, row=6, columnspan=3)


#
buttonLoadGpx = tk.Button(root, text="Load Gpx", command=LoadGpxFile)
buttonLoadGpx.grid(column=7, row=7, columnspan=3)




canvas.focus_set()


root.bind('<Configure>', ResizeCanvas)

#root.wm_iconbitmap('@icon.xbm')

root.update()

SetViewToBoundingBox(west = nwLon, north = nwLat, east = seLon, south = seLat)
LoadTiles()

##########
root.mainloop()
