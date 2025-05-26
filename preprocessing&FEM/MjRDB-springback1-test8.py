# -*- coding: mbcs -*-
import math
from abaqus import *
from abaqusConstants import *
from caeModules import *
import csv
import os
os.chdir(r"G:/mrdb-test8")  # 修改工作目录，用于最后保存odb文件
filePath = "G:/mrdb-test8"  # cvs文件路径
fr = open(filePath + "/MRDB-bending1.csv", 'r')  # csv文件名
reader = csv.reader(fr)
paralist = list(reader)
print(len(paralist))
PARALISTindex = range(len(paralist) - 3)
print(len(PARALISTindex))
numberOfBending=1
#for i in PARALISTindex:
for i in range (100):
    numofball = int(paralist[i + 3][11])  # 芯球个数
    numberOfBending = int(paralist[i + 3][35])  # 第n弯曲段
    #openMdb(pathName='F:/ABAQUS-batch/'+ str(PARALISTindex))
    openMdb(pathName=filePath + '/job'+ str(i + 1)+'-bending'+str(numberOfBending))
    #: 打开模型数据库cae.
    mdb.Model(name='Model-0'+'-Copy', objectToCopy=mdb.models['Model-job'+ str(i + 1)+'-bending'+str(numberOfBending)])
    #: 模型 "Model-0-Copy" 已创建.
    # 抑制模具实例
    a = mdb.models['Model-0-Copy'].rootAssembly
    a.deleteFeatures(('bending-die-1', 'clamp-die-1','pressure-die-1', 'wiper-die-1', 'insert-die-1',))
    for j in range(numofball):
        a.deleteFeatures(('Ball-'+str(j+1),))
    #删除原有模具部件
    del mdb.models['Model-0-Copy'].parts['Ball']
    del mdb.models['Model-0-Copy'].parts['bending-die']
    del mdb.models['Model-0-Copy'].parts['clamp-die']
    del mdb.models['Model-0-Copy'].parts['insert-die']
    del mdb.models['Model-0-Copy'].parts['pressure-die']
    #del mdb.models['Model-0-Copy'].parts['shank']
    del mdb.models['Model-0-Copy'].parts['wiper-die']
    # 删除相互作用
    del mdb.models['Model-0'+'-Copy'].interactions['general-contact1']
    for j in range(numofball):
        del mdb.models['Model-0-Copy'].interactions['Ball-' + str(j + 1)]
    #创建分析步
    del mdb.models['Model-0-Copy'].steps['Step-bending']
    mdb.models['Model-0-Copy'].StaticStep(name='Step-springback',
                                          previous='Initial', timePeriod=120, initialInc=1e-06, minInc=1e-09,
                                          maxInc=20.0, nlgeom=ON)
    mdb.models['Model-0-Copy'].steps['Step-springback'].setValues(
        stabilizationMagnitude=0.0002, stabilizationMethod=DAMPING_FACTOR,
        continueDampingFactors=False, adaptiveDampingRatio=None,maxNumInc=1000,initialInc=1e-06,
        minInc=1e-09)
    mdb.models['Model-0-Copy'].fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'SEQUT', 'PE', 'PEEQ', 'PEMAG', 'LE', 'TE', 'TEEQ',
                   'TEVOL', 'EEQUT', 'U', 'UR', 'RF', 'CF', 'CSTRESS', 'CDISP', 'STH',
                   'COORD', 'MVF'))
    mdb.models['Model-0-Copy'].steps['Step-springback'].control.setValues(
        allowPropagation=OFF, resetDefaultValues=OFF, timeIncrementation=(4.0, 8.0,
                                                                      9.0, 16.0, 10.0, 4.0, 24.0, 20.0, 6.0, 3.0, 50.0))
    #创建相互作用
    a = mdb.models['Model-0-Copy'].rootAssembly
    region1 = a.instances['shank'].surfaces['shank']
    a = mdb.models['Model-0-Copy'].rootAssembly
    region2 = a.surfaces['pipe-inside']
    mdb.models['Model-0-Copy'].SurfaceToSurfaceContactStd(
        name='shank', createStepName='Initial', master=region1, slave=region2,
        sliding=FINITE, thickness=ON, interactionProperty='mandrel',
        adjustMethod=NONE, initialClearance=OMIT, datumAxis=None,
        clearanceRegion=None)
    #: 相互作用 "shank" 已创建.


    #删除连接截面
    del mdb.models['Model-0-Copy'].sections['Beam']
    del mdb.models['Model-0-Copy'].sections['joint']
    #删除连接集
    for j in range (numofball+2):
        mdb.models['Model-0-Copy'].rootAssembly.deleteSets(setNames=(
            'Wire-'+str(j+1)+'-Set-1',))
    #删除幅值
    del mdb.models['Model-0-Copy'].amplitudes['Amp-1']
    #删除表面集
    mdb.models['Model-0-Copy'].rootAssembly.deleteSurfaces(surfaceNames=(
        'bending', 'clamp', 'insert', 'pipe', 'pressure', 'wiper',
    ))
    # mdb.models['Model-0-Copy'].rootAssembly.deleteSurfaces(surfaceNames=(
    #     'bending', 'clamp', 'insert', 'pipe', 'pipe-inside', 'pressure', 'wiper',
    # ))
    #删除约束
    for j in range(numofball):
        del mdb.models['Model-0-Copy'].constraints['Constraint-'+str(j+1)]

    #设置重启动输出
    mdb.models['Model-0-Copy'].steps['Step-springback'].Restart(
        frequency=0, numberIntervals=1, overlay=ON, timeMarks=OFF)
    #删除模具边界条件
    del mdb.models['Model-0-Copy'].boundaryConditions['pressure']
    #del mdb.models['Model-0-Copy'].boundaryConditions['shank']
    del mdb.models['Model-0-Copy'].boundaryConditions['wiper']
    del mdb.models['Model-0-Copy'].boundaryConditions['pipetail']
    #删除连接截面
    for j in range(numofball+2):
        del mdb.models['Model-0-Copy'].rootAssembly.sectionAssignments[0]
    #删除惯性
    del mdb.models['Model-0'+'-Copy'].rootAssembly.engineeringFeatures.inertias['Inertia-1']
    del mdb.models['Model-0'+'-Copy'].rootAssembly.engineeringFeatures.inertias['Inertia-2']
    del mdb.models['Model-0'+'-Copy'].rootAssembly.engineeringFeatures.inertias['Inertia-3']
    del mdb.models['Model-0'+'-Copy'].rootAssembly.engineeringFeatures.inertias['Inertia-4']
    #添加管端完全固定
    a = mdb.models['Model-0-Copy'].rootAssembly
    f1 = a.instances['pipe-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#4 ]',), )
    region = regionToolset.Region(faces=faces1)
    mdb.models['Model-0-Copy'].EncastreBC(name='BC-springback',
                                          createStepName='Initial', region=region, localCsys=None)
    #导入弯管结果
    a = mdb.models['Model-0' + '-Copy'].rootAssembly
    instances = (mdb.models['Model-0-Copy'].rootAssembly.instances['pipe-1'],)
    mdb.models['Model-0-Copy'].InitialState(updateReferenceConfiguration=ON,
                                            fileName='job'+str(i+1)+'-bending'+str(numberOfBending), endStep=LAST_STEP, endIncrement=STEP_END,
                                            name='Predefined Field-1', createStepName='Initial', instances=instances)
    mdb.models.changeKey(fromName='Model-0-Copy',
                         toName='Model-' + paralist[i + 3][0] + '-springback' + str(numberOfBending))  # 修改模型树中模型名
    #创建job
    mdb.Job(name='job'+str(i+1)+'-springback'+str(numberOfBending), model='Model-' + paralist[i + 3][0] + '-springback' + str(numberOfBending), description='', type=ANALYSIS,
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1,numDomains=1,
        numGPUs=0)
    # 写入输入文件
    mdb.jobs[paralist[i + 3][0]+'-springback'+str(numberOfBending)].writeInput(consistencyChecking=OFF)
    #提交
    #mdb.jobs['Job-26SB'].submit(consistencyChecking=OFF)
    mdb.saveAs(pathName='G:/mrdb-test8/' + paralist[i + 3][0]+'-springback'+str(numberOfBending))#保存
print ('End of programm')


