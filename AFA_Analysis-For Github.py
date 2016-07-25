# Import the tools used in the analysis
import numpy
import scipy
import pylab
import matplotlib
# This code is for use with data gathered with Robert Huber's Tracker program
#GLOBAL VARIABLES ARE BELOW EXISTING NUMBERS ARE FROM A SPECIFIC EXAMPLE. USE REFERENCE FRAMES TO DETERMINE ON YOUR OWN
pixelspercm= 16.5
Ycritical= 163.0
Xcritical= 288.0
Inner_radius= 50.0
Outer_radius= 100.0

def commandMenu():
    # This is the command menu list
    
    while True:
        
        print("\t  -----Menu ----- ")
        print()
        print("\t 0. Exit Program")
        print("\t 1. Load Data File")
        print("\t 2. Load Live Data File, and send summary stats to excel")
        print("\t 3. Load Recorded Data File, and send summary stats to excel")
        print("\t 4. summary stats to excel")
        print("\t 5. Quality Control Tools")
        print("\t 6.  save to text file")
        print("\t 7.  Look at Feeding incident")
        # Below is for recorded videos
        print("\t 8.  Look at Recorded Feeding incident")
        try:
            option = int(input("\n Enter Command Number: "))

            if option   == 0:
                return
            elif option == 1:
                data = getInput()
                return data
            elif option == 2:
                data = getInput()
                mastamatfile=createmastermatrix(data)
                tofile(mastamatfile)
            elif option == 3:
                # this is used for videos the lab recorded
                data = getInput()
                trimmed_data = trim_frames_from_list(data)
                trimmed_data=trimmed_data[3:]
                mastamatfile=createmastermatrix(trimmed_data)
                tofile(mastamatfile)
            elif option == 4:
                mastamatfile=createmastermatrix(data)
                tofile(mastamatfile)
            elif option == 5:
                data = getInput()
                trimmed_data = trim_frames_from_list(data)
                velvsr(trimmed_data)
            elif option == 6:
                data = getInput()
                trimmed_data = trim_frames_from_list(data)
                to_text_file(trimmed_data)
            elif option == 7:
                data = getInput()
                specify_feeding_incident(data)
            elif option == 8:
                data = getInput()
                trimmed_data = trim_frames_from_list(data)
                trimmed_data=trimmed_data[3:]
                specify_feeding_incident(trimmed_data)
    
        except Exception as e:
            print(e)

#below are the two different data loading functions
        
def getInput():
    import numpy as np
    # This function asks the user for an input file.
    print " must type exactly with extension and be within single parenthesis"
    filename=input("Type file name:")
    # Imports the timestamp as well as position
    newlymadearray=np.genfromtxt(filename, dtype=float, usecols=(0,1,2))
    numpy.shape(newlymadearray)
    return newlymadearray

def modgetInput():
    '''
    Same as getInput() except it only uses the X and Y coordinates and not the timestamp
    '''
    import numpy as np
    # This function asks the user for an input file.
    print " must type exactly with extension and be within single parenthesis"
    filename=input("Type file name:")
    # Imports the timestamp as well as position
    newlymadearray=np.genfromtxt(filename, dtype=float, usecols=(1,2))
    return numpy.array(newlymadearray)
    
def trim_frames_from_list(frames_list):
    '''
    When recorded videos are used there are 30 fps, however our specific live tracking analysis uses 10 fps, so this takes the inputnfrom the recorded video and takes every third frame so we can use 10fps
    '''
	n=0
	import numpy as np
	trimmed_frames_list=[[],[],[]]
	while n < (len(frames_list)-1):
		trimmed_frames_list=trimmed_frames_list+[frames_list[n]]
		n=n+3
	return trimmed_frames_list

#above is data loading

# Below is for general summary stats

def simpledistance(a,b,c,d):
    '''
    '''
    xdist = float(b) - float(a)
    ydist = float(d) - float(c)
    coorddist = numpy.sqrt(((float(ydist)**2+float(xdist)**2)))
    if coorddist < 1.1:
        #Below 1.1 means that the
        coorddist = 0.0
    return coorddist


def cart2rhofly(x,y):
    '''
    Convert a cartesian coordinate pair to a rho value
    '''
    import numpy as np
    xco=float(x)
    yco=float(y)
    newxco=float(xco-Xcritical)
    newyco=float(yco-Ycritical)
    rho = np.sqrt(newxco**2 + newyco**2)
    return rho

def cartesian_bias(loadedlist):
    '''
    Function that determines the proportion of the test a fly spends in each radial zone
    '''
    polarflylist=convert_to_polar(loadedlist)
    inner_zone=0
    middle_zone=0
    outer_zone=0
    for fly in polarflylist:
        #each item in polarflylist is a data point and belwo checks which zone it falls in
        if fly[1] <= Inner_radius:
            inner_zone = inner_zone +1
        elif fly[1] >= Outer_radius:
            outer_zone = outer_zone +1
        else:
            middle_zone = middle_zone + 1

    if inner_zone == 0:
        # avoid a divide by zero
        inner = 0
    if inner_zone > 0:
        inner=(float(inner_zone)/float(len(loadedlist)))
    if middle_zone == 0:
        # avoid a divide by zero
        middle = 0
    if middle_zone > 0:
        middle = (float(middle_zone)/float(len(loadedlist)))
    if outer_zone == 0:
        # avoid a divide by zero
        outer = 0
    if outer_zone > 0:
        outer=(float(outer_zone)/float(len(loadedlist)))
    formasterarray=[]
    # creates the readinggs to add in
    reading1=[['Innerzone',inner]]
    reading2=[['middlezone',middle]]
    reading3=[['outerzone',outer]]
    formasterarray=formasterarray+reading1
    formasterarray=formasterarray+reading2
    formasterarray=formasterarray+reading3
    #retrusn the data and feeds it into the master file
    return formasterarray


def biasfunction(loadedlist):
    '''
        This function determines the preference of a fly for a certain quadrant of the arena by determining the proportion of the number of readings in that area
        '''
    # Initialize zone counts
    Zone1 = 0
    Zone2 = 0
    Zone3 = 0
    Zone4 = 0
    #works through each entry (time, x, y) and compares the values to the 'critical values' that define the respective quadrants
    for fly in loadedlist:
        if float(fly[1]) >= Xcritical and float(fly[2]) <= Ycritical:
            Zone1 = Zone1 + 1
        if float(fly[1]) >= Xcritical and float(fly[2]) > Ycritical:
            Zone2 = Zone2 + 1
        if float(fly[1]) < Xcritical and float(fly[2]) > Ycritical:
            Zone3 = Zone3 + 1
        if float(fly[1]) < Xcritical and float(fly[2]) <= Ycritical:
            Zone4 = Zone4 + 1
    foroutput=[]
    # calculates proportions
    firstzone=(float(Zone1))/(float(len(loadedlist)))
    secondzone=(float(Zone2))/(float(len(loadedlist)))
    thirdzone=(float(Zone3))/(float(len(loadedlist)))
    fourthzone=(float(Zone4))/(float(len(loadedlist)))
    reading1=[['zone1',firstzone]]
    reading2=[['zone2',secondzone]]
    reading3=[['zone3',thirdzone]]
    reading4=[['zone4',fourthzone]]
    # adds the values to a list for output
    foroutput=foroutput+reading1
    foroutput=foroutput+reading2
    foroutput=foroutput+reading3
    foroutput=foroutput+reading4
    return foroutput

def distperpointfive(dataset):
    '''
    Determines proportion of time spent in each 0.5cm zone by the fly ding test
    '''
    forparsing=convert_to_polar(dataset)
    totalvalue=float(len(forparsing))
    # initialize cuonts
    rone=0
    rtwo=0
    rthree=0
    rfour=0
    rfive=0
    rsix=0
    rseven=0
    reight=0
    rnine=0
    rten=0
    releven=0
    rtwelve=0
    rthirteen=0
    rfourteen=0
    rfifteen=0
    rsixteen=0
    rseventeen=0
    reighteen=0
    #slot items in to the specific half cm zones
    for item in forparsing:
        itemr=((item[1])/float(pixelspercm))
        if float(itemr) <= 0.5:
            rone=rone+1
        elif float(itemr) <= 1.0:
            rtwo=rtwo+1
        elif float(itemr) <= 1.5:
            rthree=rthree+1
        elif float(itemr) <= 2.0:
            rfour=rfour+1
        elif float(itemr) <= 2.5:
            rfive=rfive+1
        elif float(itemr) <= 3.0:
            rsix=rsix+1
        elif float(itemr) <= 3.5:
            rseven=rseven+1
        elif float(itemr) <= 4.0:
            reight=reight+1
        elif float(itemr) <= 4.5:
            rnine=rnine+1
        elif float(itemr) <= 5.0:
            rten=rten+1
        elif float(itemr) <= 5.5:
            releven=releven+1
        elif float(itemr) <= 6.0:
            rtwelve=rtwelve+1
        elif float(itemr) <= 6.5:
            rthirteen=rthirteen+1
        elif float(itemr) <= 7.0:
            rfourteen=rfourteen+1
        elif float(itemr) <= 7.5:
            rfifteen=rfifteen+1
        elif float(itemr) <= 8.0:
            rsixteen=rsixteen+1
        elif float(itemr) <= 8.5:
            rseventeen=rseventeen+1
        else:
            reighteen=reighteen+1
        # Add prorportions to master file
        reading1=[['0.0-0.5cm',(rone/totalvalue)]]
        reading2=[['0.5-1cm',(rtwo/totalvalue)]]
        reading3=[['1.0-1.5cm',(rthree/totalvalue)]]
        reading4=[['1.5-2.0cm',(rfour/totalvalue)]]
        reading5=[['2.0-2.5cm',(rfive/totalvalue)]]
        reading6=[['2.5-3.0cm',(rsix/totalvalue)]]
        reading7=[['3.0-3.5cm',(rseven/totalvalue)]]
        reading8=[['3.5-4.0cm',(reight/totalvalue)]]
        reading9=[['4.0-4.5cm',(rnine/totalvalue)]]
        reading10=[['4.5-5cm',(rten/totalvalue)]]
        reading11=[['5.0-5.5cm',(releven/totalvalue)]]
        reading12=[['5.5-6.0cm',(rtwelve/totalvalue)]]
        reading13=[['6.0-6.5cm',(rthirteen/totalvalue)]]
        reading14=[['6.5-7.0cm',(rfourteen/totalvalue)]]
        reading15=[['7.0-7.5cm',(rfifteen/totalvalue)]]
        reading16=[['7.5-8.0cm',(rsixteen/totalvalue)]]
        reading17=[['8.0-8.5cm',(rseventeen/totalvalue)]]
        reading18=[['8.5-9.0cm',(reighteen/totalvalue)]]
        forfinalarray=[]
        forfinalarray=forfinalarray+reading1+reading2+reading3+reading4+reading5+reading6+reading7+reading8+reading9+reading10+reading11+reading12+reading13+reading14+reading15+reading16+reading17+reading18
    return forfinalarray
    #above is for general summary stats

def convert_to_polar(cartesian):
    '''
    converts cartesian coordinates to polar coordinates
    '''
    import numpy as np
    newflies=[]
    for flytime in cartesian:
        timestamp=float(flytime[0])
        xco=float(flytime[1])
        yco=float(flytime[2])
        newxco=float(xco-Xcritical)
        newyco=float(yco-Ycritical)
        rho = np.sqrt(newxco**2 + newyco**2)
        phi = np.arctan2(newyco, newxco)
        newreading=[[timestamp,rho,phi]]
        newflies=newflies+newreading
    return newflies

def averager(polarr):
    '''
Computes the averate radial value
    '''
    totalr= []
    for polarpt in polarr:
        distancevalue=float(polarpt[1])/float(pixelspercm)
        if distancevalue>9.0:
            distancevalue = 9.0
        totalr=totalr+[distancevalue]
    averagerval=numpy.array(totalr)
    sumval=numpy.sum(averagerval)
    radial=sumval/float(len(totalr))
    fornewlist=[]
    reading=[['average radial value (cm)',radial]]
    fornewlist=fornewlist+reading
    return fornewlist

def distance_total_coordinates(flydata):
    '''
        calculates
    '''
    cumulativedist=0
    reading=0
    indivDist = []
    while reading < (len(flydata)-1):
        correcteddist=0
        ##need to make it use the next reading
        x2=int(flydata[int(reading+1)][1])
        x1=int(flydata[int(reading)][1])
        y2=int(flydata[int(reading+1)][2])
        y1=int(flydata[int(reading)][2])
        coordinatedist=simpledistance(x1,x2,y1,y2)
        indivDist.append([float(coordinatedist)/float(pixelspercm)])
        reading = reading + 1
    return indivDist

def cumulativedist(distanceinfo):
    '''
    '''
    cumulativedistance=distance_total_coordinates(distanceinfo)
    sumdist=numpy.sum(cumulativedistance)
    fornewlist=[]
    reading= [['Distance traveled (cm)', sumdist]]
    fornewlist=fornewlist+reading
    return fornewlist

def distntime_total_coordinates(flydata):
    cumulativedist=0
    reading=0
    disttimelist = []
    while reading < (len(flydata)-1):
        correcteddist=0
        time1=int(flydata[int(reading)][0])
        time2=int(flydata[int(reading+1)][0])
        x2=int(flydata[int(reading+1)][1])
        x1=int(flydata[int(reading)][1])
        y2=int(flydata[int(reading+1)][2])
        y1=int(flydata[int(reading)][2])
        coordinatedist=simpledistance(x1,x2,y1,y2)
        time=float((time2-time1)/float(1000)) #converts time to seconds
        correcteddist=([float(coordinatedist)/float(pixelspercm)])
        correctdistarray=numpy.array(correcteddist)
        reading = reading + 1
        pointvel=float(float(correctdistarray)/float(time))
        disttimelist=disttimelist + [pointvel]
    disttimearray=numpy.array(disttimelist)
    averagevel=(numpy.sum(disttimearray)/float(len(disttimearray)))
    fornewlist=[]
    newpoint = [['velocity during test (cm/sec)',averagevel]]
    return newpoint

def distntime_zonevelocity(flydata):
    cumulativedist=0
    flylength=12 #define value
    bodylenthresh=0.5 # can define value
    #pixelspercm=50 #please define value
    reading=0
    #flylength=12 #define value #720
    #pixelspercm=37 #please define value #720
    disttimelist = []
    while reading < (len(flydata)-1):
        correcteddist=0
        time1=int(flydata[int(reading)][0])
        time2=int(flydata[int(reading+1)][0])
        x2=int(flydata[int(reading+1)][1])
        x1=int(flydata[int(reading)][1])
        y2=int(flydata[int(reading+1)][2])
        y1=int(flydata[int(reading)][2])
        coordinatedist=simpledistance(x1,x2,y1,y2)
        time=float((time2-time1)/float(1000)) #converts time to seconds
        correcteddist=([float(coordinatedist)/float(pixelspercm)])
        correctdistarray=numpy.array(correcteddist)
        reading = reading + 1
        pointvel=float(float(correctdistarray)/float(time))
        disttimelist=disttimelist + [pointvel]
    disttimearray=numpy.array(disttimelist)
    totalvalue=float(len(disttimearray))
    velone=0
    veltwo=0
    velthree=0
    velfour=0
    velfive=0
    velsix=0
    velseven=0
    veleight=0
    velnine=0
    for item in disttimearray:
        if float(item)< 0.5:
            velone=velone+1
        elif float(item)< 1.0:
            veltwo=veltwo+1
        elif float(item)< 1.5:
            velthree=velthree+1
        elif float(item)< 2.0:
            velfour=velfour+1
        elif float(item)< 2.5:
            velfive=velfive+1
        elif float(item)< 3.0:
            velsix=velsix+1
        elif float(item)< 3.5:
            velseven=velseven+1
        elif float(item)< 4.0:
            veleight=veleight+1
        else:
            velnine=velnine+1
    read1=[['0.0-0.5cm/sec',(velone/totalvalue)]]
    read2=[['0.5-1cm/sec',(veltwo/totalvalue)]]
    read3=[['1.0-1.5cm/sec',(velthree/totalvalue)]]
    read4=[['1.5-2.0cm/sec',(velfour/totalvalue)]]
    read5=[['2.0-2.5cm/sec',(velfive/totalvalue)]]
    read6=[['2.5-3.0cm/sec',(velsix/totalvalue)]]
    read7=[['3.0-3.5cm/sec',(velseven/totalvalue)]]
    read8=[['3.5-4.0cm/sec',(veleight/totalvalue)]]
    read9=[['4.0+cm/sec',(velnine/totalvalue)]]
    tosend=[]
    tosend=tosend+read1+read2+read3+read4+read5+read6+read7+read8+read9
    return tosend

def velineachradialzone(importfile):
    reading=0
    radialvel=[]
    while reading < (len(importfile)-1):
        time1=int(importfile[int(reading)][0])
        time2=int(importfile[int(reading+1)][0])
        x2=int(importfile[int(reading+1)][1])
        x1=int(importfile[int(reading)][1])
        y2=int(importfile[int(reading+1)][2])
        y1=int(importfile[int(reading)][2])
        coordinatedist=simpledistance(x1,x2,y1,y2)
        time=float((time2-time1)/float(1000))
        correcteddist=(float(coordinatedist)/float(pixelspercm))
        correctdistarray=numpy.array(correcteddist)
        velocityforreading=float(correctdistarray)/float(time)
        rhoforreading=cart2rhofly(int(importfile[int(reading)][1]),int(importfile[int(reading)][2]))
        radialvel=radialvel+[[rhoforreading,velocityforreading]]
        reading=reading+1
    innercount=0
    middlecount=0
    outercount=0
    vrinner=[]
    vrmiddle=[]
    vrouter=[]
    for velnr in radialvel:
        if velnr[0] < Inner_radius:
            vrinner=vrinner+[float(velnr[1])]
            innercount=innercount+1
        elif velnr[0] > Outer_radius:
            vrouter=vrouter+[float(velnr[1])]
            outercount=outercount+1
        else:
            vrmiddle=vrmiddle+[float(velnr[1])]
            middlecount=middlecount+1
    innerarray=numpy.array(vrinner)
    middlearray=numpy.array(vrmiddle)
    outerarray=numpy.array(vrouter)
    tosend=[]
    read1=[['inner zone velocity (cm/sec)',(numpy.sum(innerarray)/float(innercount))]]
    read2=[['middle zone velocity (cm/sec)',(numpy.sum(middlearray)/float(middlecount))]]
    read3=[['outer zone velocity (cm/sec)',(numpy.sum(outerarray)/float(outercount))]]
    tosend=tosend+read1+read2+read3
    return tosend

def velvsr(importfile):
    reading=0
    radialvel=[]
    maxvel=0.0
    veltime=0.0
    while reading < (len(importfile)-1):
        time1=int(importfile[int(reading)][0])
        time2=int(importfile[int(reading+1)][0])
        x2=int(importfile[int(reading+1)][1])
        x1=int(importfile[int(reading)][1])
        y2=int(importfile[int(reading+1)][2])
        y1=int(importfile[int(reading)][2])
        coordinatedist=simpledistance(x1,x2,y1,y2)
        time=float((time2-time1)/float(1000))
        correcteddist=(float(coordinatedist)/float(pixelspercm))
        correctdistarray=numpy.array(correcteddist)
        velocityforreading=float(correctdistarray)/float(time)
        if velocityforreading > maxvel:
            veltime=time1
            maxvel=velocityforreading
            print veltime
        rhoforreading=cart2rhofly(int(importfile[int(reading)][1]),int(importfile[int(reading)][2]))
        nurho=(float(rhoforreading)/float(pixelspercm))
        if nurho >9.0:
            nurho=9.0
        radialvel=radialvel+[[nurho,velocityforreading]]
        reading=reading+1
    radialvelarray=numpy.array(radialvel)
    print veltime
    import  matplotlib.pyplot as plt
    x, y = radialvelarray[:,0], radialvelarray[:,1]
    heatmap, xedges, yedges = plt.hexbin(x, y, bins=2)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap, extent=extent)
    plt.show()



def local(desiredlist):
    import numpy as np
    newflies=[]
    reading=0
    paramx=[0][1]
    paramy=[0][2]
    while reading < (len(desiredlist)-1):
        timestamp=float(flytime[0])
        xco=float(flytime[reading][1])
        yco=float(flytime[reading][2])
        simpledistance(paramx,xco,paramy,yco)
        correcteddist=simpledistance/float(pixelspercm)
        newreading=[[timestamp,correcteddist]]
        newflies=newflies+newreading
        reading=reading+1
    print newflies

def local_search_plot(desiredlist):
    reading=0
    #pixelspercm=50
    startingpoint=0
    disttime=[]
    while reading < (len(desiredlist)-1):
        correcteddist=0
        timeread=float(reading[0])
        ##need to make it use the next reading
        x2=int(desiredlist[int(reading+1)][1])
        x1=int(desiredlist[int(startingpoint)][1])
        y2=int(desiredlist[int(reading+1)][2])
        y1=int(desiredlist[int(startingpoint)][2])
        coordinatedist=simpledistance(x1,x2,y1,y2)
        print coordinatedist
        fixeddist=(float(coordinatedist)/float(pixelspercm))
        nuread=[fixeddist,timeread]
        disttime=disttime+nuread
        reading = reading + 1
    disttimearray=numpy.array(disttime)
    
    print disttimearray


##the problem is below
def local_search(desiredlist):
    reading=0
    indivDist=0
    print "start read"
    while reading < (len(desiredlist)-1):
        correcteddist=0
        ##need to make it use the next reading
        x2=float(desiredlist[int(reading+1)][1])
        x1=float(desiredlist[0][1])
        y2=float(desiredlist[int(reading+1)][2])
        y1=float(desiredlist[0][2])
        coordinatedist=simpledistance(x1,x2,y1,y2)
        print coordinatedist
        normalizeddist=([float(coordinatedist)/float(pixelspercm)])
        indivDist=indivDist+normalizeddist
        reading = reading + 1
    avgrafter=(float(indivDist)/float(reading))
    read1=[['average distance from drop (cm)', avgrafter]]
    print read1
    forlist=forlist+read1
    timevsdist=[]
    while reading < (len(desiredlist)-1):
        correcteddist=0
        ##need to make it use the next reading
        time=float(desiredlist[int(reading)])
        x2=float(desiredlist[int(reading+1)][1])
        x1=float(desiredlist[0][1])
        y2=float(desiredlist[int(reading+1)][2])
        y1=float(desiredlist[0][2])
        coordinatedist=simpledistance(x1,x2,y1,y2)
        indivDist=indivDist+([float(coordinatedist)/float(pixelspercm)])
        newreadagain=[[time,indivDist]]
        forlist=forlist+[newreadagain]
        reading = reading + 1
    return forlist

def Distance_from_drop_post(feedfile):
    reading=0
    distance=0
    totality=len(feedfile)
    while reading < (len(feedfile)-1):
        origx=int(feedfile[0][1])
        origy=int(feedfile[0][2])
        newx=int(feedfile[reading][1])
        newy=int(feedfile[reading][2])
        dropdist=simpledistance(origx,newx,origy,newy)
        actualdist=dropdist/pixelspercm
        distance=distance+actualdist
        reading=reading+1
    avgd=distance/float(totality)
    print "distance post (cm)"
    print avgd # get this to be written to a file

def Distance_from_drop_pre(feedfile):
    reading=0
    distance=0
    totality=len(feedfile)
    while reading < (len(feedfile)-1):
        origx=int(feedfile[-1][1])
        origy=int(feedfile[-1][2])
        newx=int(feedfile[reading][1])
        newy=int(feedfile[reading][2])
        dropdist=simpledistance(origx,newx,origy,newy)
        actualdist=dropdist/pixelspercm
        distance=distance+actualdist
        reading=reading+1
    avgd=distance/float(totality)
    read=[['avg distance pre (cm)',avgd]]
    return read

def local_search_line(search_activity):
    seconds=1
    counter=0
    initial_time=search_activity[0][0]
    search_list=[]
    search_list=search_list+[['time since feeding','distance (cm)']]
    initialdropx=search_activity[0][1]
    initialdropy=search_activity[0][2]
    timegoal=float((float(initial_time)+1000))
    for element in search_activity:
        if seconds < 30:
            if float(element[0]) >= timegoal:
                readx=element[1]
                ready=element[2]
                reading_distance=simpledistance(initialdropx,readx,initialdropy,ready)
                search_distance=reading_distance/float(pixelspercm)
                search_read=[[seconds, search_distance]]
                search_list=search_list+search_read
                timegoal=float(timegoal+1000)
                seconds=seconds+1
                counter=counter+1
    final_dist=simpledistance(initialdropx,search_activity[-1][1],initialdropy,search_activity[-1][2])
    finaldsearch=final_dist/float(pixelspercm)
    final_read=[[30, finaldsearch]]
    search_list=search_list+final_read
    return search_list

def specify_feeding_incident(fullfile):
    #currently set up to do 30 seconds pre and post
    feedingtime=int(input("Type timestamp feeding ends:"))
    feedingduration=int(input("how long did it feed for:"))
    pre_ingestive=[]
    post_ingestive=[]
    after_feeding=[]
    before_feeding=[]
    for point in fullfile:
        if float(point[0]) < float(feedingtime):
            before_feeding=before_feeding+[point]
        else:
            after_feeding=after_feeding+[point]
    for prefood in before_feeding:
        if prefood[0] > float((feedingtime-(feedingduration*1000))-30000):
            if prefood[0] < float(feedingtime-(feedingduration*1000)):
                pre_ingestive=pre_ingestive+[prefood]
    for postfood in after_feeding:
        if postfood[0] < float(feedingtime+30000):
            post_ingestive=post_ingestive+[postfood]
    thirty_after_line=local_search_line(post_ingestive)
    print "save dist from drop every sec for 30 sec"
    tofile(thirty_after_line)
    preingestive_behaviour=createmastermatrix(pre_ingestive)
    print "save pre feeding"
    tofile(preingestive_behaviour)
    postingestive_behaviour=createmastermatrix(post_ingestive)
    print "save post feeding"
    tofile(postingestive_behaviour)
    print "save distance summary"
    avgdropdistprenpost=[]
    avgdropdistprenpost=avgdropdistprenpost+[Distance_from_drop_pre(pre_ingestive)]
    avgdropdistprenpost[Distance_from_drop_post(post_ingestive)]
    tofile(avgdropdistprenpost)

              
def createmastermatrix(filetoworkwith):
    '''
    Function calls other functions that compute statistics on the data set, and compiles them all into a single matrix which can then be sent to a file by tofile()
        
    '''
    mastermatrix=[]
    name = raw_input("Title of test: ")
    firstheader=[['parameter',name]]
    # add a header to make the file easy to understand when written
    mastermatrix=mastermatrix+firstheader
    # computes parameters one by one for the test and then adds them to the list
    radialzones=cartesian_bias(filetoworkwith)
    mastermatrix=mastermatrix+radialzones
    quadrantzones=biasfunction(filetoworkwith)
    mastermatrix=mastermatrix+quadrantzones
    Timeinbinradial=distperpointfive(filetoworkwith)
    mastermatrix=mastermatrix+Timeinbinradial
    averageradiallist=convert_to_polar(filetoworkwith)
    averageradial=averager(averageradiallist)
    mastermatrix=mastermatrix+averageradial
    cumdist=cumulativedist(filetoworkwith)
    mastermatrix=mastermatrix+cumdist
    totaltestaveragevel=distntime_total_coordinates(filetoworkwith)
    mastermatrix=mastermatrix+totaltestaveragevel
    binnedvelocity=distntime_zonevelocity(filetoworkwith)
    mastermatrix=mastermatrix+binnedvelocity
    velineachzone=velineachradialzone(filetoworkwith)
    mastermatrix=mastermatrix+velineachzone
    
    return mastermatrix


def to_text_file(matrix):
    # where name can be inserted to auto write file without user input
    import os
    name = raw_input("Name your output file: ")
    outputDataName = name
    directory = os.getcwd()
    outputDataName = directory + "/" + outputDataName + ".txt"
    # writes the file where we specified
    fl = open('outputDataName', 'w')
    for values in matrix:
        fl.write(str(values) + '\n')
    print ("file can be found in"+" "+directory)
    fl.close()


def tofile(matrix):
    # where name can be inserted to auto write file without user input
    import os
    name = raw_input("Name your output file: ")
    outputDataName = name
    directory = os.getcwd()
    outputDataName = directory + "/" + outputDataName + ".csv"
    import csv
    # writes the file where we specified
    fl = open(outputDataName, 'w')
    writer = csv.writer(fl)
    for values in matrix:
        writer.writerow(values)
    print ("file can be found in"+" "+directory)
    fl.close()


def runmodAFA_ANALYSIS():
    
    try:
        commandMenu()
    except Exception as inst:
        print("-------------")
        input(inst)

# -----------
# Entry Point
# -----------
#cd desktop

runmodAFA_ANALYSIS()
		
