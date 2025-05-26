# -*- coding: mbcs -*-
import math
from abaqus import *
from abaqusConstants import *
from caeModules import *
import csv
import os





##############################################################################
#####说明：本程序为通过在.csv文件中设置参数，批量完成建模、接触、边界等设置，最终完成批量前处理
##############################################################################

os.chdir(r"G:/mrdb-test8")#修改工作目录，用于最后保存odb文件
filePath = "G:/mrdb-test8/"#cvs文件路径
fr = open(filePath+"MRDB-bending1.csv",'r')#csv文件名
reader = csv.reader(fr)
paralist=list(reader)
print(len(paralist))
PARALISTindex=range(len(paralist)-3)
print(len(PARALISTindex))


#for i in PARALISTindex:
for i in range(100):
    Mdb()#: 新的模型数据库已创建.
    mdb.models.changeKey(fromName='Model-1', toName='Model-0')
    diameter=float(paralist[i+3][1])#管材外径
    thickness=float(paralist[i+3][2])#壳模型壁厚
    radius=float(paralist[i+3][3])#弯曲半径
    angleofbending=float(paralist[i+3][4])#弯曲角度
    angularvelocity=float(paralist[i+3][5])#弯曲角速度
    timeofbending=(angleofbending*math.pi/180)/angularvelocity
    speedratioofpressure=float(paralist[i+3][6])#辅推速比
    speedratioofboost=float(paralist[i+3][7])#助推速比
    clamp_force=float(paralist[i+3][8])#夹紧模压力
    pressure_force=float(paralist[i+3][9])#压模压力
    boost_force=float(paralist[i+3][10])#助推模压力
    numofball=int(paralist[i+3][11]) # 芯球个数
    prolongationofshank=float(paralist[i+3][12]) # 芯轴初始伸长量
    initialpositionofpressure=float(paralist[i+3][13]) # 压模初始位置
    frictionofbending=float(paralist[i+3][14])#管材与弯曲模摩擦系数fb
    frictionofpressure=float(paralist[i+3][15])#管材与压模摩擦系数fp
    frictionofwiper=float(paralist[i+3][16])#管材与防皱模摩擦系数fw
    frictionofclamp=float(paralist[i+3][17])#管材与夹紧模摩擦系数fc
    frictionofmandrel=float(paralist[i+3][18])#管材与芯棒摩擦系数fc
    gapofbending=float(paralist[i+3][19])#弯曲模间隙（0-0.25）
    gapofclamp = float(paralist[i+3][20])  # 夹紧模间隙（0.1）
    gapofpressure=float(paralist[i+3][21])#压模间隙（0-0.25）
    gapofwiper=float(paralist[i+3][22])#防皱模间隙（0-0.25）
    diameterofdie1=float(paralist[i+3][23])#模具直径过盈量
    diameterofdie=diameter+2*diameterofdie1
    lengthofclamp=float(paralist[i+3][24])
    #lengthofpressure=float(paralist[i+3][25])#压模长度
    lengthofpressure = 2 * math.pi * radius * angleofbending / 360 * speedratioofpressure + 150
    lengthofwiper=float(paralist[i+3][26])#防皱模长度
    diameterofmandrel = float(paralist[i+3][27])  # 芯棒直径
    lengthofshank=float(paralist[i+3][28]) # 芯轴长度
    filletofshank=float(paralist[i+3][29])#芯轴圆角
    widthofmandrel=float(paralist[i+3][30]) # 芯棒宽度
    spacingofmandrel=float(paralist[i+3][31]) # 芯棒间距
    massscaling=float(paralist[i+3][32])  # 质量缩放系数
    feedLength=float(paralist[i+3][33])  # 进给长度
    torsionAngle=float(paralist[i+3][34])  # 扭转角度
    numberOfBending=int(paralist[i+3][35])  # 第n弯曲段
    widthofbending=diameter+30
    lengthofpipe=angularvelocity*timeofbending*radius+lengthofclamp+3000
    boostervelocityofpressure=angularvelocity*speedratioofpressure*radius
    boostervelocityofboost = angularvelocity * speedratioofboost * radius
    frictionofgeneral=0.05

    #########################
    ####### 模型建立 #########
    #########################
    #1.1弯曲模建模
    s = mdb.models['Model-0'].ConstrainedSketch(name='__profile__',
        sheetSize=1000.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
    s.FixedConstraint(entity=g[2])
    s.ArcByCenterEnds(center=(radius, 0.0), point1=(radius, -(diameter/2.0)), point2=(radius,
        diameter/2.0), direction=CLOCKWISE)#半圆弧
    s.Line(point1=(radius, (diameter/2.0)), point2=(radius, widthofbending/2.0))
    s.VerticalConstraint(entity=g[4], addUndoState=False)#线段
    s.Line(point1=(radius, -(diameter/2.0)), point2=(radius, -(widthofbending)/2.0))
    s.VerticalConstraint(entity=g[5], addUndoState=False)#线段
    p = mdb.models['Model-0'].Part(name='bending-die', dimensionality=THREE_D,
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-0'].parts['bending-die']
    p.BaseShellRevolve(sketch=s, angle=360.0, flipRevolveDirection=OFF)#旋转拉伸
    s.unsetPrimaryObject()
    del mdb.models['Model-0'].sketches['__profile__']
    p.ReferencePoint(point=(0.0, 0.0, 0.0))#选择刚体参考点
    mdb.models['Model-0'].parts['bending-die'].features.changeKey(fromName='RP',
        toName='RP-bending')#重命名参考点
    r = p.referencePoints
    refPoints=(r[2], )
    p.Set(referencePoints=refPoints, name='bending-die')#建立集合
    #1.2管件建模
    s = mdb.models['Model-0'].ConstrainedSketch(name='__profile__',
                                                sheetSize=100.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(diameter/2, 0.0))
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(diameter/2-thickness, 0.0))
    p = mdb.models['Model-0'].Part(name='pipe', dimensionality=THREE_D,
                                   type=DEFORMABLE_BODY)
    p = mdb.models['Model-0'].parts['pipe']
    p.BaseSolidExtrude(sketch=s, depth=lengthofpipe)
    s.unsetPrimaryObject()
    p = mdb.models['Model-0'].parts['pipe']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['Model-0'].sketches['__profile__']
    p = mdb.models['Model-0'].parts['pipe']
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#1 ]',), )
    p.Surface(side1Faces=side1Faces, name='pipe')
    #: 表面 'pipe' 已创建 (1 面).
    #1.3夹紧模
    s = mdb.models['Model-0'].ConstrainedSketch(name='__profile__',
        sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
    s.FixedConstraint(entity=g[2])
    s.ArcByCenterEnds(center=(radius, 0.0), point1=(radius, -(diameter/2.0)), point2=(radius,
        diameter/2.0), direction=COUNTERCLOCKWISE)#半圆弧
    s.Line(point1=(radius, (diameter/2.0)), point2=(radius, widthofbending/2.0))#线段
    s.VerticalConstraint(entity=g[4], addUndoState=False)
    s.Line(point1=(radius, -(diameter/2.0)), point2=(radius, -(widthofbending)/2.0))#线段
    s.VerticalConstraint(entity=g[5], addUndoState=False)
    p = mdb.models['Model-0'].Part(name='clamp-die', dimensionality=THREE_D,
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-0'].parts['clamp-die']
    p.BaseShellExtrude(sketch=s, depth=lengthofclamp)#拉伸
    s.unsetPrimaryObject()
    del mdb.models['Model-0'].sketches['__profile__']
    v1, e, d2, n = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=p.InterestingPoint(edge=e[4], rule=CENTER))#选择刚体参考点
    mdb.models['Model-0'].parts['clamp-die'].features.changeKey(fromName='RP',
        toName='RP-clamp')#参考点重命名
    r = p.referencePoints
    refPoints=(r[2], )
    p.Set(referencePoints=refPoints, name='clamp-die')#建立集合
    #1.4压模
    s1 = mdb.models['Model-0'].ConstrainedSketch(name='__profile__',
        sheetSize=200.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.ArcByCenterEnds(center=(0.0, 0.0), point1=(0.0, -(diameter/2.0)), point2=(0.0, diameter/2.0),
        direction=CLOCKWISE)#半圆弧
    p = mdb.models['Model-0'].Part(name='pressure-die', dimensionality=THREE_D,
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-0'].parts['pressure-die']
    p.BaseShellExtrude(sketch=s1, depth=lengthofpressure)#拉伸
    s1.unsetPrimaryObject()
    del mdb.models['Model-0'].sketches['__profile__']
    v2, e1, d1, n1 = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=p.InterestingPoint(edge=e1[0], rule=CENTER))#选择刚体参考点
    mdb.models['Model-0'].parts['pressure-die'].features.changeKey(fromName='RP',
        toName='RP-pressure')#重命名参考点
    r = p.referencePoints
    refPoints=(r[2], )
    p.Set(referencePoints=refPoints, name='pressure-die')#建立集合
    #1.5防皱模
    s = mdb.models['Model-0'].ConstrainedSketch(name='__profile__',
        sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.ArcByCenterEnds(center=(0.0, 0.0), point1=(0.0, -(diameter/2.0)), point2=(0.0, diameter/2.0),
        direction=CLOCKWISE)#半圆弧
    p = mdb.models['Model-0'].Part(name='wiper-die', dimensionality=THREE_D,
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-0'].parts['wiper-die']
    p.BaseShellExtrude(sketch=s, depth=lengthofwiper)#拉伸
    s.unsetPrimaryObject()
    del mdb.models['Model-0'].sketches['__profile__']
    v1, e, d2, n = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=p.InterestingPoint(edge=e[2], rule=CENTER))#选择刚体参考点
    mdb.models['Model-0'].parts['wiper-die'].features.changeKey(fromName='RP',
        toName='RP-wiper')#参考点重命名
    r = p.referencePoints
    refPoints=(r[2], )
    p.Set(referencePoints=refPoints, name='wiper-die')#建立集合
    #1.6镶块建模
    s = mdb.models['Model-0'].ConstrainedSketch(name='__profile__',
        sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.ArcByCenterEnds(center=(radius, 0.0), point1=(radius, -(diameter/2.0)), point2=(radius,
        diameter/2.0), direction=CLOCKWISE)#半圆弧
    p = mdb.models['Model-0'].Part(name='insert-die', dimensionality=THREE_D,
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-0'].parts['insert-die']
    p.BaseShellExtrude(sketch=s, depth=lengthofclamp)#拉伸
    s.unsetPrimaryObject()
    del mdb.models['Model-0'].sketches['__profile__']
    v1, e, d2, n = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=p.InterestingPoint(edge=e[0], rule=CENTER))#选择刚体参考点
    mdb.models['Model-0'].parts['insert-die'].features.changeKey(fromName='RP',
        toName='RP-insert')#参考点重命名
    r = p.referencePoints
    refPoints=(r[2], )
    p.Set(referencePoints=refPoints, name='insert-die')#建立集合
    #芯棒
    s1 = mdb.models['Model-0'].ConstrainedSketch(name='__profile__',
                                                    sheetSize=2000.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -1000.0), point2=(0.0, 1000.0))
    s1.FixedConstraint(entity=g[2])
    s1.ConstructionLine(point1=(widthofmandrel/2, 0.0), angle=90.0)
    s1.VerticalConstraint(entity=g[3], addUndoState=False)
    s1.ConstructionLine(point1=(-(widthofmandrel/2), 0.0), angle=90.0)
    s1.VerticalConstraint(entity=g[4], addUndoState=False)
    s1.ConstructionLine(point1=(0.0, 0.0), angle=0.0)
    s1.HorizontalConstraint(entity=g[5], addUndoState=False)
    s1.ConstructionCircleByCenterPerimeter(center=(0.0, 0.0), point1=(diameterofmandrel/2, 0.0))
    s1.Line(point1=(widthofmandrel/2, math.sqrt(math.pow((diameterofmandrel/2),2)-math.pow(widthofmandrel/2,2) )), point2=(widthofmandrel/2, 0.0))
    s1.VerticalConstraint(entity=g[7], addUndoState=False)
    s1.ParallelConstraint(entity1=g[3], entity2=g[7], addUndoState=False)
    s1.CoincidentConstraint(entity1=v[0], entity2=g[3], addUndoState=False)
    s1.CoincidentConstraint(entity1=v[1], entity2=g[3], addUndoState=False)
    s1.Line(point1=(widthofmandrel/2, 0.0), point2=(-(widthofmandrel/2), 0.0))
    s1.HorizontalConstraint(entity=g[8], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[7], entity2=g[8], addUndoState=False)
    s1.CoincidentConstraint(entity1=v[2], entity2=g[4], addUndoState=False)
    s1.Line(point1=(-widthofmandrel/2, 0.0), point2=(-(widthofmandrel/2), math.sqrt(math.pow((diameterofmandrel/2),2)-math.pow(widthofmandrel/2,2) )))
    s1.VerticalConstraint(entity=g[9], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[8], entity2=g[9], addUndoState=False)
    s1.CoincidentConstraint(entity1=v[3], entity2=g[4], addUndoState=False)
    s1.ArcByCenterEnds(center=(0.0, 0.0), point1=(-(widthofmandrel/2), math.sqrt(math.pow((diameterofmandrel/2),2)-math.pow(widthofmandrel/2,2) )), point2=(
        widthofmandrel/2, math.sqrt(math.pow((diameterofmandrel/2),2)-math.pow(widthofmandrel/2,2) )), direction=CLOCKWISE)
    s1.CoincidentConstraint(entity1=v[4], entity2=g[2], addUndoState=False)
    s1.sketchOptions.setValues(constructionGeometry=ON)
    s1.assignCenterline(line=g[5])
    p = mdb.models['Model-0'].Part(name='Ball', dimensionality=THREE_D,
                                      type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-0'].parts['Ball']
    p.BaseSolidRevolve(sketch=s1, angle=360, flipRevolveDirection=OFF)
    s1.unsetPrimaryObject()
    p = mdb.models['Model-0'].parts['Ball']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['Model-0'].sketches['__profile__']
    p = mdb.models['Model-0'].parts['Ball']
    c1 = p.cells
    p.RemoveCells(cellList = c1[0:1])
    p = mdb.models['Model-0'].parts['Ball']
    p.ReferencePoint(point=(0.0, 0.0, 0.0))
    mdb.models['Model-0'].parts['Ball'].features.changeKey(fromName='RP', toName='RP-Ball')
    p = mdb.models['Model-0'].parts['Ball']
    r = p.referencePoints
    refPoints = (r[3],)
    p.Set(referencePoints=refPoints, name='RP-Ball')
    p = mdb.models['Model-0'].parts['Ball']
    region=p.sets['RP-Ball']
    mdb.models['Model-0'].parts['Ball'].engineeringFeatures.PointMassInertia(
        name='Ball', region=region, mass=1e-08, i11=0.00, i22=0.00, i33=0.00,
    alpha=0.0, composite=0.0)
    #创建芯球表面
    p = mdb.models['Model-0'].parts['Ball']
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#4 ]', ), )
    p.Surface(side1Faces=side1Faces, name='Ball')
    #: The surface 'ball' has been edited (1 face).
    # 芯球网络划分
    p = mdb.models['Model-0'].parts['Ball']
    p.seedPart(size=(diameter * 3.1415926) / 40, deviationFactor=0.025, minSizeFactor=0.15)
    p.generateMesh()
    #芯轴
    s1 = mdb.models['Model-0'].ConstrainedSketch(name='__profile__',
                                                 sheetSize=2000.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -1000.0), point2=(0.0, 1000.0))
    s1.FixedConstraint(entity=g[2])
    s1.Line(point1=(0.0, 0.0), point2=(0.0, lengthofshank))
    s1.VerticalConstraint(entity=g[3], addUndoState=False)
    s1.ParallelConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
    s1.CoincidentConstraint(entity1=v[0], entity2=g[2], addUndoState=False)
    s1.CoincidentConstraint(entity1=v[1], entity2=g[2], addUndoState=False)
    s1.Line(point1=(0.0, lengthofshank), point2=(-diameterofmandrel/2.0, lengthofshank))
    s1.HorizontalConstraint(entity=g[4], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s1.Line(point1=(-diameterofmandrel/2.0, lengthofshank), point2=(-diameterofmandrel/2.0, 0.0))
    s1.VerticalConstraint(entity=g[5], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    s1.Line(point1=(-diameterofmandrel/2.0, 0.0), point2=(0.0, 0.0))
    s1.HorizontalConstraint(entity=g[6], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[5], entity2=g[6], addUndoState=False)
    p = mdb.models['Model-0'].Part(name='shank', dimensionality=THREE_D,
                                   type=DEFORMABLE_BODY)
    p = mdb.models['Model-0'].parts['shank']
    p.BaseSolidRevolve(sketch=s1, angle=360.0, flipRevolveDirection=OFF)
    s1.unsetPrimaryObject()
    p = mdb.models['Model-0'].parts['shank']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['Model-0'].sketches['__profile__']
    p = mdb.models['Model-0'].parts['shank']
    p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=diameterofmandrel / 8)
    p = mdb.models['Model-0'].parts['shank']
    d = p.datums
    t = p.MakeSketchTransform(sketchPlane=d[2], sketchUpEdge=d[1],
                              sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=( 0.0, widthofmandrel/2, diameterofmandrel / 8))
    s = mdb.models['Model-0'].ConstrainedSketch(name='__profile__',
                                                sheetSize=279.22, gridSpacing=6.98, transform=t)
    g, v, d1, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=SUPERIMPOSE)
    p = mdb.models['Model-0'].parts['shank']
    p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
    s.ArcByCenterEnds(center=(0.0, 0.0), point1=(-widthofmandrel / 4, widthofmandrel / 4), point2=(widthofmandrel / 4, widthofmandrel / 4), direction=COUNTERCLOCKWISE)
    s.Line(point1=(widthofmandrel / 4, widthofmandrel / 4), point2=(widthofmandrel ,0.8*widthofmandrel))
    # s.PerpendicularConstraint(entity1=g[9], entity2=g[10], addUndoState=False)
    s.Line(point1=(widthofmandrel ,0.8*widthofmandrel), point2=(-widthofmandrel ,0.8*widthofmandrel))
    # s.HorizontalConstraint(entity=g[11], addUndoState=False)
    s.Line(point1=(-widthofmandrel ,0.8*widthofmandrel), point2=(-widthofmandrel / 4, widthofmandrel / 4))
    p = mdb.models['Model-0'].parts['shank']
    d1 = p.datums
    p.CutExtrude(sketchPlane=d1[2], sketchUpEdge=d1[1], sketchPlaneSide=SIDE1,
                 sketchOrientation=RIGHT, sketch=s, depth=diameterofmandrel / 4, flipExtrudeDirection=OFF)
    s.unsetPrimaryObject()
    del mdb.models['Model-0'].sketches['__profile__']
    p = mdb.models['Model-0'].parts['shank']
    e = p.edges
    p.Round(radius=filletofshank, edgeList=(e[14],))
    p1 = mdb.models['Model-0'].parts['shank']
    session.viewports['Viewport: 1'].setValues(displayedObject=p1)
    p = mdb.models['Model-0'].parts['shank']
    p.ReferencePoint(point=( 0.0,widthofmandrel/2, 0.0))
    mdb.models['Model-0'].parts['shank'].features.changeKey(fromName='RP',toName='RP-shank')
    p = mdb.models['Model-0'].parts['shank']
    r = p.referencePoints
    refPoints = (r[5],)
    p.Set(referencePoints=refPoints, name='shank')
    #: 集 'shank' 已创建 (1 参考点).
    p = mdb.models['Model-0'].parts['shank']
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#3 ]',), )
    p.Surface(side1Faces=side1Faces, name='shank')
    mdb.models['Model-0'].parts['shank'].setValues(space=THREE_D,
                                                   type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-0'].parts['shank']
    c1 = p.cells
    p.RemoveCells(cellList=c1[0:1])
    # 2.1定义材料
    # mdb.models['Model-0'].Material(name='copper-T1')
    # mdb.models['Model-0'].materials['copper-T1'].Elastic(table=((110000,
    #     0.32), ))
    # mdb.models['Model-0'].materials['copper-T1'].Density(table=((8.9e-09, ), ))
    # mdb.models['Model-0'].materials['copper-T1'].Conductivity(table=((390, ), ))
    # mdb.models['Model-0'].materials['copper-T1'].Plastic(table=((240.0, 0.0), (245.0,
    #     0.01), (250.0, 0.02), (260.0, 0.03), (263.0, 0.04), (
    #     267.0, 0.05), (270.0, 0.06), (267.0, 0.07), (263.0, 0.08),
    #     (260.0, 0.09),(255.0, 0.1)))
    #
    # mdb.models['Model-0'].Material(name='SI_mm113111_6061-T6(GB)')
    # mdb.models['Model-0'].materials['SI_mm113111_6061-T6(GB)'].Elastic(table=((69000.0006661372,
    #                                                                            0.33),))
    # mdb.models['Model-0'].materials['SI_mm113111_6061-T6(GB)'].Density(table=((2.7e-09,),))
    # mdb.models['Model-0'].materials['SI_mm113111_6061-T6(GB)'].Conductivity(table=((166.9,),))
    # mdb.models['Model-0'].materials['SI_mm113111_6061-T6(GB)'].Plastic(table=((275.0, 0.0), (275.0,
    #                                                                                          0.004), (275.79029, 0.01),
    #                                                                           (277.16924, 0.015), (282.68505, 0.02), (
    #                                                                               296.47456, 0.03), (310.26408, 0.04),
    #                                                                           (324.05359, 0.06), (344.73786, 0.08),
    #                                                                           (355.76948, 0.09)))
    # from material import createMaterialFromDataString
    # createMaterialFromDataString('Model-0', 'SI_mm304', '2016',
    #                              """{'specificHeat': {'temperatureDependency': OFF, 'table': ((460000000.0,),), 'dependencies': 0, 'law': CONSTANTVOLUME}, 'materialIdentifier': '', 'description': '', 'elastic': {'temperatureDependency': OFF, 'moduli': LONG_TERM, 'noCompression': OFF, 'noTension': OFF, 'dependencies': 0, 'table': ((190000.0, 0.29),), 'type': ISOTROPIC}, 'density': {'temperatureDependency': OFF, 'table': ((7.93e-09,),), 'dependencies': 0, 'fieldName': '', 'distributionType': UNIFORM}, 'name': 'SI_mm304', 'plastic': {'temperatureDependency': OFF, 'strainRangeDependency': OFF, 'rate': OFF, 'dependencies': 0, 'hardening': ISOTROPIC, 'dataType': HALF_CYCLE, 'table': ((206.807, 0.0), (215.0, 0.0011), (303.6, 0.1176), (376.0, 0.234), (432.5, 0.35), (472.8, 0.47), (479.0, 0.58), (505.0, 0.7)), 'numBackstresses': 1}, 'expansion': {'temperatureDependency': OFF, 'userSubroutine': OFF, 'zero': 0.0, 'dependencies': 0, 'table': ((1.8e-05,),), 'type': ISOTROPIC}, 'conductivity': {'temperatureDependency': OFF, 'table': ((16.3,),), 'dependencies': 0, 'type': ISOTROPIC}}""")
    # #: Material 'SI_mm304' has been copied to the current model.
    mdb.models['Model-0'].Material(name='Si_mm316L')
    mdb.models['Model-0'].materials['Si_mm316L'].Density(table=((
                                                                                7.8e-09,),))
    mdb.models['Model-0'].materials['Si_mm316L'].Elastic(table=((
                                                                                114220.0, 0.28),))
    mdb.models['Model-0'].materials['Si_mm316L'].Plastic(
        hardening=ISOTROPIC, table=((234.7028, 0.0), (304.65128, 0.00382), (
            351.81364, 0.02531), (423.58245, 0.06096), (532.55387, 0.1225), (628.34327,
                                                                             0.19365), (727.64787, 0.28709),
                                    (830.17474, 0.39682), (844.82144, 0.45463),
                                    (813.77044, 0.46651)))
    #2.2定义截面
    mdb.models['Model-0'].HomogeneousSolidSection(name='Section-1',
                                                  material='Si_mm316L', thickness=None)
    #2.3截面指派
    p = mdb.models['Model-0'].parts['pipe']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['Model-0'].parts['pipe']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]',), )
    region = regionToolset.Region(cells=cells)
    p = mdb.models['Model-0'].parts['pipe']
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    #3组合装配
    #3.1零件实例化
    a = mdb.models['Model-0'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-0'].parts['bending-die']
    a.Instance(name='bending-die-1', part=p, dependent=ON)
    p = mdb.models['Model-0'].parts['clamp-die']
    a.Instance(name='clamp-die-1', part=p, dependent=ON)
    p = mdb.models['Model-0'].parts['pipe']
    a.Instance(name='pipe-1', part=p, dependent=ON)
    p = mdb.models['Model-0'].parts['pressure-die']
    a.Instance(name='pressure-die-1', part=p, dependent=ON)
    p = mdb.models['Model-0'].parts['wiper-die']
    a.Instance(name='wiper-die-1', part=p, dependent=ON)
    p = mdb.models['Model-0'].parts['insert-die']
    a.Instance(name='insert-die-1', part=p, dependent=ON)
    a = mdb.models['Model-0'].rootAssembly
    p = mdb.models['Model-0'].parts['shank']
    a.Instance(name='shank', part=p, dependent=ON)
    for j in range(numofball):
        a = mdb.models['Model-0'].rootAssembly
        p = mdb.models['Model-0'].parts['Ball']
        a.Instance(name='Ball-'+str(j+1), part=p, dependent=ON)
        # mdb.models['Model-0'].rootAssembly.features.changeKey(
        #     fromName='adjustment-1-rad-' + str(j + 2), toName='adjustment-' + str(j + 2))
    #3.2旋转操作
    a = mdb.models['Model-0'].rootAssembly
    a.rotate(instanceList=('bending-die-1', 'clamp-die-1','insert-die-1',), axisPoint=(0.0, 0.0, 0.0),
        axisDirection=(0.0, 0.0, 1.0), angle=90.0)
    a.rotate(instanceList=('clamp-die-1','insert-die-1',), axisPoint=(0.0, 0.0, 0.0),
        axisDirection=(0.0, 1.0, 0.0), angle=180.0)
    a.rotate(instanceList=('pipe-1', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(
        0.0, 1.0, 0.0), angle=180.0)
    a.rotate(instanceList=('pressure-die-1', ), axisPoint=(0.0, 0.0, 0.0),
        axisDirection=(0.0, 0.0, -1.0), angle=90.0)
    a.rotate(instanceList=('wiper-die-1', ), axisPoint=(0.0, 0.0, 0.0),
        axisDirection=(0.0, 0.0, 1.0), angle=90.0)
    a.rotate(instanceList=('wiper-die-1',), axisPoint=(0.0, 0.0, 0.0),
             axisDirection=(0.0, 1.0, 0.0), angle=180.0)
    a.rotate(instanceList=('shank',), axisPoint=(0.0, 0.0, 0.0), axisDirection=(1.0, 0.0, 0.0), angle=-90.0)
    a.rotate(instanceList=('shank',), axisPoint=(0.0, 0.0, 0.0), axisDirection=(0.0, 0.0, 1.0), angle=90.0)
    #3.3平移操作
    a = mdb.models['Model-0'].rootAssembly
    a.translate(instanceList=('pipe-1',), vector=(0.0, 0.0, 0.0))
    a.translate(instanceList=('bending-die-1',), vector=(0.0, -(radius + gapofbending), -(feedLength)))
    a.translate(instanceList=('pressure-die-1',),
                vector=(0.0, gapofpressure, -(lengthofpressure + initialpositionofpressure+feedLength)))
    a.translate(instanceList=('clamp-die-1',), vector=(0.0, -(radius - gapofclamp), -feedLength+lengthofclamp))
    a.translate(instanceList=('insert-die-1',), vector=(0.0, -(radius + gapofbending), -feedLength+lengthofclamp))
    a.translate(instanceList=('shank',), vector=(0.0, 0.0, -(feedLength-prolongationofshank)))
    a.translate(instanceList=('wiper-die-1',), vector=(0.0, -gapofwiper, -(feedLength+5)))
    #芯球调整
    for j in range(numofball):
        a = mdb.models['Model-0'].rootAssembly
        a.rotate(instanceList=('Ball-'+str(j+1),), axisPoint=(
            0.0, 0.0, 0.0), axisDirection=(0.0, 1.0, 0.0), angle=90.0)
        a.rotate(instanceList=('Ball-' + str(j+1),), axisPoint=(
            0.0, 0.0, 0.0), axisDirection=(0.0, 0.0, 1.0), angle=90.0)
        a.translate(instanceList=('Ball-'+str(j+1),),
                    vector=(0.0, 0.0, (j+1)*spacingofmandrel-(widthofmandrel/2)+prolongationofshank-feedLength))
    #4定义分析步
    mdb.models['Model-0'].ExplicitDynamicsStep(name='Step-bending', previous='Initial',
    timePeriod=timeofbending, massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, massscaling,
    0.0, None, 0, 0, 0.0, 0.0, 0, None), ))#质量缩放
    #重启动
    mdb.models['Model-0'].steps['Step-bending'].Restart(numberIntervals=1,
                                                           overlay=ON, timeMarks=OFF)
    #定义场输出
    mdb.models['Model-0'].fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'SVAVG', 'SEQUT', 'PE', 'PEVAVG', 'PEEQ', 'PEEQVAVG', 'LE',
                   'TE', 'TEEQ', 'TEVOL', 'EEQUT', 'U', 'UR','V', 'A', 'RF', 'CSTRESS', 'EVF',
                   'STH', 'COORD', 'MVF'))
    ############
    # 5定义相互作用
    # 5.1定义表面
    a = mdb.models['Model-0'].rootAssembly
    s1 = a.instances['pipe-1'].faces
    side1Faces1 = s1.getSequenceFromMask(mask=('[#1 ]',), )
    a.Surface(side1Faces=side1Faces1, name='pipe')
    a = mdb.models['Model-0'].rootAssembly
    s1 = a.instances['pipe-1'].faces
    side2Faces1 = s1.getSequenceFromMask(mask=('[#2 ]',), )
    a.Surface(side2Faces=side2Faces1, name='pipe-inside')
    #: 表面 'pipe-inside' 已创建 (2 面).
    s1 = a.instances['bending-die-1'].faces
    side1Faces1 = s1.getSequenceFromMask(mask=('[#2 ]',), )
    a.Surface(side1Faces=side1Faces1, name='bending')  # 创建表面
    s1 = a.instances['clamp-die-1'].faces
    side2Faces1 = s1.getSequenceFromMask(mask=('[#2 ]',), )
    a.Surface(side2Faces=side2Faces1, name='clamp')  # 创建表面
    s1 = a.instances['pressure-die-1'].faces
    side1Faces1 = s1.getSequenceFromMask(mask=('[#1 ]',), )
    a.Surface(side1Faces=side1Faces1, name='pressure')  # 创建表面
    s1 = a.instances['wiper-die-1'].faces
    side1Faces1 = s1.getSequenceFromMask(mask=('[#1 ]',), )
    a.Surface(side1Faces=side1Faces1, name='wiper')  # 创建表面
    s1 = a.instances['insert-die-1'].faces
    side1Faces1 = s1.getSequenceFromMask(mask=('[#1 ]',), )
    a.Surface(side1Faces=side1Faces1, name='insert')  # 创建表面
    a = mdb.models['Model-0'].rootAssembly
    s1 = a.instances['shank'].faces
    side1Faces1 = s1.getSequenceFromMask(mask=('[#3 ]',), )
    a.Surface(side1Faces=side1Faces1, name='shank')
    #: 表面 'shank' 已创建 (2 面).
    #
    # 5.2定义接触属性
    mdb.models['Model-0'].ContactProperty('bending')
    mdb.models['Model-0'].interactionProperties['bending'].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=(( frictionofbending,),),
        shearStressLimit=None, maximumElasticSlip=FRACTION,
        fraction=0.005, elasticSlipStiffness=None)
    mdb.models['Model-0'].interactionProperties['bending'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)
    # bending接触属性
    mdb.models['Model-0'].ContactProperty('pressure')
    mdb.models['Model-0'].interactionProperties['pressure'].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((frictionofpressure,),),
        shearStressLimit=None, maximumElasticSlip=FRACTION,
        fraction=0.005, elasticSlipStiffness=None)
    mdb.models['Model-0'].interactionProperties['pressure'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)
    # pressure接触属性
    mdb.models['Model-0'].ContactProperty('wiper')
    mdb.models['Model-0'].interactionProperties['wiper'].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((frictionofwiper,),),
        shearStressLimit=None, maximumElasticSlip=FRACTION,
        fraction=0.005, elasticSlipStiffness=None)
    mdb.models['Model-0'].interactionProperties['wiper'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)
    # wiper接触属性
    mdb.models['Model-0'].ContactProperty('clamp')
    mdb.models['Model-0'].interactionProperties['clamp'].TangentialBehavior(
        formulation=ROUGH)
    mdb.models['Model-0'].interactionProperties['clamp'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT)
    mdb.models['Model-0'].interactionProperties['clamp'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)
    # clamp接触属性
    mdb.models['Model-0'].ContactProperty('general')
    mdb.models['Model-0'].interactionProperties['general'].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((frictionofgeneral,),),
        shearStressLimit=None, maximumElasticSlip=FRACTION,
        fraction=0.005, elasticSlipStiffness=None)
    mdb.models['Model-0'].interactionProperties['general'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)
    mdb.models['Model-0'].ContactProperty('mandrel')
    mdb.models['Model-0'].interactionProperties['mandrel'].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=(( 0.1,),), shearStressLimit=None,
        maximumElasticSlip=FRACTION,
        fraction=0.005, elasticSlipStiffness=None)
    mdb.models['Model-0'].interactionProperties['mandrel'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT)
    mdb.models['Model-0'].ContactProperty('free')
    # 5.3定义接触
    # mdb.models['Model-0'].ContactExp(name='general-contact1',
    #                                              createStepName='Initial')
    # r11 = mdb.models['Model-0'].rootAssembly.surfaces['bending']
    # r12 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    # r21 = mdb.models['Model-0'].rootAssembly.surfaces['clamp']
    # r22 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    # r31 = mdb.models['Model-0'].rootAssembly.surfaces['insert']
    # r32 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    # r41 = mdb.models['Model-0'].rootAssembly.surfaces['pressure']
    # r42 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    # r51 = mdb.models['Model-0'].rootAssembly.surfaces['wiper']
    # r52 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    # r61 = mdb.models['Model-0'].rootAssembly.instances['Ball-1'].surfaces['Ball']
    # r62 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    # r71 = mdb.models['Model-0'].rootAssembly.instances['Ball-2'].surfaces['Ball']
    # r72 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    # r81 = mdb.models['Model-0'].rootAssembly.instances['Ball-3'].surfaces['Ball']
    # r82 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    # r91 = mdb.models['Model-0'].rootAssembly.instances['Ball-4'].surfaces['Ball']
    # r92 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    # r101 = mdb.models['Model-0'].rootAssembly.instances['shank'].surfaces['shank']
    # r102 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    # mdb.models['Model-0'].interactions['general-contact1'].includedPairs.setValuesInStep(
    #     stepName='Initial', useAllstar=OFF, addPairs=((r11, r12), (r21, r22), (r31,
    #                                                                            r32), (r41, r42), (r51, r52), (r61, r62),
    #                                                   (r71, r72), (r81, r82), (r91,
    #                                                                            r92), (r101, r102)))
    # r21 = mdb.models['Model-0'].rootAssembly.surfaces['clamp']
    # r22 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    # r31 = mdb.models['Model-0'].rootAssembly.surfaces['insert']
    # r32 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    # r41 = mdb.models['Model-0'].rootAssembly.surfaces['pressure']
    # r42 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    # r51 = mdb.models['Model-0'].rootAssembly.surfaces['wiper']
    # r52 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    # r61 = mdb.models['Model-0'].rootAssembly.instances['Ball-1'].surfaces['Ball']
    # r62 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    # r71 = mdb.models['Model-0'].rootAssembly.instances['Ball-2'].surfaces['Ball']
    # r72 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    # r81 = mdb.models['Model-0'].rootAssembly.instances['Ball-3'].surfaces['Ball']
    # r82 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    # r91 = mdb.models['Model-0'].rootAssembly.instances['Ball-4'].surfaces['Ball']
    # r92 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    # r101 = mdb.models['Model-0'].rootAssembly.instances['shank'].surfaces['shank']
    # r102 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    # mdb.models['Model-0'].interactions['general-contact1'].contactPropertyAssignments.appendInStep(
    #     stepName='Initial', assignments=((GLOBAL, SELF, 'bending'), (r21, r22,
    #                                                                  'clamp'), (r31, r32, 'clamp'),
    #                                      (r41, r42, 'pressure'), (r51, r52, 'wiper'),
    #                                      (r61, r62, 'mandrel'), (r71, r72, 'mandrel'), (r81, r82, 'mandrel'), (r91,
    #                                                                                                            r92,
    #                                                                                                            'mandrel'),
    #                                      (r101, r102, 'mandrel')))
    # #: 相互作用 "general-contact1" 已创建.
    mdb.models['Model-0'].ContactExp(name='general-contact1', createStepName='Initial')

    # 静态定义的接触面
    static_pairs = [
        ('bending', 'pipe'),
        ('clamp', 'pipe'),
        ('insert', 'pipe'),
        ('pressure', 'pipe'),
        ('wiper', 'pipe'),
        ('shank', 'pipe-inside')
    ]

    # 动态定义的球接触面
    ball_pairs = [('Ball-%d' % (j + 1), 'pipe-inside') for j in range(numofball)]

    # 处理静态接触面对
    for pair_name in static_pairs:
        r1 = mdb.models['Model-0'].rootAssembly.surfaces[pair_name[0]]
        r2 = mdb.models['Model-0'].rootAssembly.surfaces[pair_name[1]]
        mdb.models['Model-0'].interactions['general-contact1'].includedPairs.setValuesInStep(
            stepName='Initial', useAllstar=OFF, addPairs=((r1, r2),)
        )

    # # 处理球体实例的接触面对
    # for jj in range(numofball):
    #     instance_name = 'Ball-%d' % (jj + 1)
    #     surface_name = 'Ball'
    #     r1 = mdb.models['Model-0'].rootAssembly.instances[instance_name].surfaces[surface_name]
    #     r2 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    #     mdb.models['Model-0'].interactions['general-contact1'].includedPairs.setValuesInStep(
    #         stepName='Initial', useAllstar=OFF, addPairs=((r1, r2),)
    #     )
    # 分配接触属性
    r21 = mdb.models['Model-0'].rootAssembly.surfaces['clamp']
    r22 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    r31 = mdb.models['Model-0'].rootAssembly.surfaces['insert']
    r32 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    r41 = mdb.models['Model-0'].rootAssembly.surfaces['pressure']
    r42 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    r51 = mdb.models['Model-0'].rootAssembly.surfaces['wiper']
    r52 = mdb.models['Model-0'].rootAssembly.surfaces['pipe']
    r61 = mdb.models['Model-0'].rootAssembly.instances['shank'].surfaces['shank']
    r62 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    mdb.models['Model-0'].interactions['general-contact1'].contactPropertyAssignments.appendInStep(
        stepName='Initial', assignments=((GLOBAL, SELF, 'bending'),
                                         (r21, r22, 'clamp'),
                                         (r31, r32, 'clamp'),
                                         (r41, r42, 'pressure'),
                                         (r51, r52, 'wiper'),
                                         (r61, r62, 'mandrel'), ))
    # for jj in range(numofball):
    #     instance_name = 'Ball-%d' % (jj + 1)
    #     surface_name = 'Ball'
    #     r1 = mdb.models['Model-0'].rootAssembly.instances[instance_name].surfaces[surface_name]
    #     r2 = mdb.models['Model-0'].rootAssembly.surfaces['pipe-inside']
    #     mdb.models['Model-0'].interactions['general-contact1'].contactPropertyAssignments.appendInStep(
    #          stepName='Initial', assignments=((r1, r2, 'mandrel'),))
    mdb.models['Model-0'].interactions['general-contact1'].setValues(
        globalSmoothing=True)
    # a = mdb.models['Model-0'].rootAssembly
    # region1 = a.surfaces['bending']
    # region2 = a.surfaces['pipe']
    # mdb.models['Model-0'].SurfaceToSurfaceContactExp(name='bending',
    #                                                  createStepName='Initial', master=region1, slave=region2,
    #                                                  mechanicalConstraint=PENALTY, sliding=FINITE,
    #                                                  interactionProperty='bending', initialClearance=OMIT,
    #                                                  datumAxis=None,
    #                                                  clearanceRegion=None)  # bending接触
    # region1 = a.surfaces['pressure']
    # region2 = a.surfaces['pipe']
    # mdb.models['Model-0'].SurfaceToSurfaceContactExp(name='pressure',
    #                                                  createStepName='Step-bending', master=region1, slave=region2,
    #                                                  mechanicalConstraint=PENALTY, sliding=FINITE,
    #                                                  interactionProperty='pressure', initialClearance=OMIT,
    #                                                  datumAxis=None,
    #                                                  clearanceRegion=None)  # pressure接触
    # region1 = a.surfaces['wiper']
    # region2 = a.surfaces['pipe']
    # mdb.models['Model-0'].SurfaceToSurfaceContactExp(name='wiper',
    #                                                  createStepName='Initial', master=region1, slave=region2,
    #                                                  mechanicalConstraint=PENALTY, sliding=FINITE,
    #                                                  interactionProperty='wiper', initialClearance=OMIT, datumAxis=None,
    #                                                  clearanceRegion=None)  # wiper接触
    # region1 = a.surfaces['clamp']
    # region2 = a.surfaces['pipe']
    # mdb.models['Model-0'].SurfaceToSurfaceContactExp(name='clamp',
    #                                                  createStepName='Step-bending', master=region1, slave=region2,
    #                                                  mechanicalConstraint=PENALTY, sliding=FINITE,
    #                                                  interactionProperty='clamp',
    #                                                  initialClearance=OMIT, datumAxis=None,
    #                                                  clearanceRegion=None)  # clamp接触
    # region1=a.surfaces['insert']
    # region2=a.surfaces['pipe']
    # mdb.models['Model-0'].SurfaceToSurfaceContactExp(name ='insert',
    #                                                 createStepName='Initial', master = region1, slave = region2,
    #                                                 mechanicalConstraint=PENALTY, sliding=FINITE,
    #                                                 interactionProperty='clamp',
    #                                                 initialClearance=OMIT, datumAxis=None,
    #                                                 clearanceRegion=None) # insert接触
    # region1 = a.instances['shank'].surfaces['shank']
    # region2 = a.surfaces['pipe-inside']
    # mdb.models['Model-0'].SurfaceToSurfaceContactExp(name='shank',
    #                                                  createStepName='Initial', master=region1, slave=region2,
    #                                                  mechanicalConstraint=PENALTY, sliding=FINITE,
    #                                                  interactionProperty='mandrel', initialClearance=OMIT,
    #                                                  datumAxis=None,
    #                                                  clearanceRegion=None) # shank接触
    for ii in range(numofball):
        a = mdb.models['Model-0'].rootAssembly
        region1 = a.instances['Ball-'+str(ii+1)].surfaces['Ball']
        a = mdb.models['Model-0'].rootAssembly
        region2 = a.surfaces['pipe-inside']
        mdb.models['Model-0'].SurfaceToSurfaceContactExp(name='Ball-'+str(ii+1),
                                                         createStepName='Initial', master=region1, slave=region2,
                                                         mechanicalConstraint=PENALTY, sliding=FINITE,
                                                         interactionProperty='mandrel', initialClearance=OMIT,
                                                         datumAxis=None,
                                                         clearanceRegion=None) # ball 接触
    # mdb.models['Model-0'].ContactProperty('free')
    # #: 相互作用属性 "free" 已创建.
    # 创建芯球间位置约束
    for j in range(numofball):
        a = mdb.models['Model-0'].rootAssembly
        if j == 0:
            region1 = a.instances['shank'].sets['shank']
        else:
            region1 = a.instances['Ball-' + str(j)].sets['RP-Ball']
        a = mdb.models['Model-0'].rootAssembly
        s1 = a.instances['Ball-' + str(j + 1)].faces
        side1Faces1 = s1.getSequenceFromMask(mask=('[#1 ]',), )
        region2 = regionToolset.Region(side1Faces=side1Faces1)
        mdb.models['Model-0'].Coupling(name='Constraint-' + str(j + 1), controlPoint=region1,
                                       surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                                       localCsys=None, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=OFF, ur3=OFF)
    # 5.5定义连接
    a = mdb.models['Model-0'].rootAssembly
    r1 = a.instances['bending-die-1'].referencePoints  # 连接点1
    r2 = a.instances['clamp-die-1'].referencePoints  # 连接点2
    a.WirePolyLine(points=((r1[2], r2[2]),), mergeType=IMPRINT, meshable=OFF)
    e1 = a.edges
    edges1 = e1.getSequenceFromMask(mask=('[#1 ]',), )
    a.Set(edges=edges1, name='Wire-1-Set-1')  # 定义线条特征
    a = mdb.models['Model-0'].rootAssembly
    r11 = a.instances['bending-die-1'].referencePoints
    r12 = a.instances['insert-die-1'].referencePoints
    a.WirePolyLine(points=((r11[2], r12[2]),), mergeType=IMPRINT, meshable=OFF)
    e1 = a.edges
    edges1 = e1.getSequenceFromMask(mask=('[#1 ]',), )
    a.Set(edges=edges1, name='Wire-2-Set-1')
    mdb.models['Model-0'].ConnectorSection(name='Beam', assembledType=BEAM)
    region = a.sets['Wire-1-Set-1']
    csa = a.SectionAssignment(sectionName='Beam', region=region)  # 定义连接截面
    region = a.sets['Wire-2-Set-1']
    csa = a.SectionAssignment(sectionName='Beam', region=region)
    # ################################################
    #     #芯棒约束设置
    # ################################################
    mdb.models['Model-0'].ConnectorSection(name='joint', translationalType=JOIN)  # 定义球铰连接
    for j in range(numofball):
        if j == 0:
            a = mdb.models['Model-0'].rootAssembly
            r13 = a.instances['Ball-1'].referencePoints
            r14 = a.instances['shank'].referencePoints
            dtm1 = a.DatumCsysByThreePoints(origin=r13[3], point1=r14[5],
                                            coordSysType=CARTESIAN)
            dtmid1 = a.datums[dtm1.id]
            a = mdb.models['Model-0'].rootAssembly
            r1 = a.instances['Ball-1'].referencePoints
            r2 = a.instances['shank'].referencePoints
            wire = a.WirePolyLine(points=((r1[3], r2[5]),), mergeType=IMPRINT,
                                  meshable=False)
        else:
            a = mdb.models['Model-0'].rootAssembly
            r1 = a.instances['Ball-'+str(j+1)].referencePoints
            r2 = a.instances['Ball-'+str(j)].referencePoints
            dtm1 = a.DatumCsysByThreePoints(origin=r1[3], point1=r2[3],
                                            coordSysType=CARTESIAN)
            dtmid1 = a.datums[dtm1.id]
            a = mdb.models['Model-0'].rootAssembly
            r1 = a.instances['Ball-'+str(j+1)].referencePoints
            r2 = a.instances['Ball-'+str(j)].referencePoints
            wire = a.WirePolyLine(points=((r1[3], r2[3]),), mergeType=IMPRINT,
                                  meshable=False)
        oldName = wire.name
        mdb.models['Model-0'].rootAssembly.features.changeKey(fromName=oldName,
                                                              toName='Wire-' + str(j + 3) + '-Set-1')
        a = mdb.models['Model-0'].rootAssembly
        e1 = a.edges
        edges1 = e1.getSequenceFromMask(mask=('[#1 ]',), )
        a.Set(edges=edges1, name='Wire-' + str(j + 3) + '-Set-1')
        region = mdb.models['Model-0'].rootAssembly.sets['Wire-' + str(j + 3) + '-Set-1']
        csa = a.SectionAssignment(sectionName='joint', region=region)
        #: 截面 "joint" 已指派给 1 条线或附着线.
        a.ConnectorOrientation(region=csa.getSet(), localCsys1=dtmid1)

    # 5.6定义惯性inertia
    a = mdb.models['Model-0'].rootAssembly
    region = a.instances['clamp-die-1'].sets['clamp-die']
    mdb.models['Model-0'].rootAssembly.engineeringFeatures.PointMassInertia(
        name='Inertia-1', region=region, mass=0.001, i11=100.0, i22=100.0,
        i33=100.0, alpha=0.0, composite=0.0)
    a = mdb.models['Model-0'].rootAssembly
    region = a.instances['insert-die-1'].sets['insert-die']
    mdb.models['Model-0'].rootAssembly.engineeringFeatures.PointMassInertia(
        name='Inertia-2', region=region, mass=0.001, i11=100.0, i22=100.0,
        i33=100.0, alpha=0.0, composite=0.0)
    a = mdb.models['Model-0'].rootAssembly
    region = a.instances['bending-die-1'].sets['bending-die']
    mdb.models['Model-0'].rootAssembly.engineeringFeatures.PointMassInertia(
        name='Inertia-3', region=region, mass=0.001, i11=100.0, i22=100.0,
        i33=100.0, alpha=0.0, composite=0.0)
    a = mdb.models['Model-0'].rootAssembly
    region = a.instances['pressure-die-1'].sets['pressure-die']
    mdb.models['Model-0'].rootAssembly.engineeringFeatures.PointMassInertia(
        name='Inertia-4', region=region, mass=0.001, i11=100.0, i22=100.0,
        i33=100.0, alpha=0.0, composite=0.0)
    #6定义边界
    mdb.models['Model-0'].SmoothStepAmplitude(name='Amp-1', timeSpan=STEP,
    data=((0.0, 0.0), (0.0001, 1.0),(1, 1.0)))
    #6.1定义防皱模边界条件
    a = mdb.models['Model-0'].rootAssembly
    region = a.instances['wiper-die-1'].sets['wiper-die']
    mdb.models['Model-0'].EncastreBC(name='wiper', createStepName='Initial',
        region=region, localCsys=None)
    #6.2定义压模边界条件
    a = mdb.models['Model-0'].rootAssembly
    region = a.instances['pressure-die-1'].sets['pressure-die']
    mdb.models['Model-0'].VelocityBC(name='pressure', createStepName='Initial',
                                        region=region, v1=0.0, v2=0.0, v3=0.0, vr1=0.0, vr2=0.0, vr3=0.0,
                                        amplitude='Amp-1', localCsys=None, distributionType=UNIFORM, fieldName='')
    mdb.models['Model-0'].boundaryConditions['pressure'].setValuesInStep(
        stepName='Step-bending', v3=boostervelocityofpressure)
    boostervelocityofpressure
    #6.3定义弯曲模边界条件
    a = mdb.models['Model-0'].rootAssembly
    region = a.instances['bending-die-1'].sets['bending-die']
    mdb.models['Model-0'].VelocityBC(name='bending', createStepName='Step-bending',
        region=region, v1=0.0, v2=0.0, v3=0.0, vr1=angularvelocity, vr2=0.0, vr3=0.0,
        amplitude='Amp-1', localCsys=None, distributionType=UNIFORM, fieldName='')
    #6.4固定芯轴
    a = mdb.models['Model-0'].rootAssembly
    region = a.instances['shank'].sets['shank']
    mdb.models['Model-0'].EncastreBC(name='shank', createStepName='Initial',
                                        region=region, localCsys=None)
    #固定管子尾端
    a = mdb.models['Model-0'].rootAssembly
    f1 = a.instances['pipe-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#4 ]',), )
    a.Set(faces=faces1, name='pipetail')
    #: 集 'pipetail' 已创建 (1 面).
    a = mdb.models['Model-0'].rootAssembly
    region = a.sets['pipetail']
    mdb.models['Model-0'].DisplacementBC(name='pipetail',
                                                     createStepName='Initial', region=region, u1=SET, u2=SET, u3=UNSET,
                                                     ur1=SET,
                                                     ur2=SET, ur3=SET, amplitude=UNSET, distributionType=UNIFORM,
                                                     fieldName='',
                                                     localCsys=None)
    # 7网格划分
    # 弯曲模网格划分
    p = mdb.models['Model-0'].parts['bending-die']
    p.seedPart(size=(diameter * 3.1415926) / 40, deviationFactor=0.01, minSizeFactor=0.1)  # 全局布种
    p.generateMesh()  # 网格划分
    # 夹紧模网格划分
    p = mdb.models['Model-0'].parts['clamp-die']
    p.seedPart(size=(diameter * 3.1415926) / 30, deviationFactor=0.01, minSizeFactor=0.1)
    p.generateMesh()
    #管件划分网格
    p = mdb.models['Model-0'].parts['pipe']
    c = p.cells
    pickedRegions = c.getSequenceFromMask(mask=('[#1 ]',), )
    p.setMeshControls(regions=pickedRegions, algorithm=MEDIAL_AXIS)
    p.seedPart(size=(diameter * 3.1415926) / 40, deviationFactor=0.018, minSizeFactor=0.1)
    p.generateMesh()
    # 压模划分网格
    p = mdb.models['Model-0'].parts['pressure-die']
    p.seedPart(size=(diameter * 3.1415926) / 40, deviationFactor=0.01, minSizeFactor=0.1)
    p.generateMesh()
    # 防皱模划分网格
    p = mdb.models['Model-0'].parts['wiper-die']
    p.seedPart(size=(diameter * 3.1415926) / 40, deviationFactor=0.01, minSizeFactor=0.1)
    p.generateMesh()
    # 镶块网格划分
    p = mdb.models['Model-0'].parts['insert-die']
    p.seedPart(size=(diameter * 3.1415926) / 30, deviationFactor=0.01, minSizeFactor=0.1)
    p.generateMesh()
    # 芯棒网络划分
    p = mdb.models['Model-0'].parts['shank']
    p.seedPart(size=(diameter * 3.1415926) / 40, deviationFactor=0.01, minSizeFactor=0.1)
    p.generateMesh()
    a = mdb.models['Model-0'].rootAssembly
    # del a.features['wiper-die-1']
    # mdb.models['Model-0'].boundaryConditions['wiper'].suppress()
    # mdb.models['Model-0'].interactions['wiper'].suppress()
    a = mdb.models['Model-0'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    a.regenerate()
    print('i final=',i)
    mdb.models.changeKey(fromName='Model-0', toName='Model-' + paralist[i + 3][0]+'-bending1')  # 修改模型树中模型名
    mdb.Model(name='Model-0', modelType=STANDARD_EXPLICIT)  # 建立新模型
    mdb.Job(name=paralist[i + 3][0]+'-bending1', model='Model-' + paralist[i + 3][0]+'-bending1', description='', type=ANALYSIS,
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
            memoryUnits=PERCENTAGE, explicitPrecision=SINGLE,
            nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF,
            contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='',
            resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=4,
            activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=2)  # 创建任务
    # mdb.jobs['bending'+paralist[i+3][0]].submit(consistencyChecking=OFF)#提交任务
    mdb.jobs[paralist[i + 3][0]+'-bending1'].writeInput(consistencyChecking=OFF)
    #: 作业输入文件已写入到 "contact-1.inp".
    del mdb.models['Model-0']
    mdb.saveAs(pathName='G:/mrdb-test8/' + paralist[i + 3][0]+'-bending1')  # mdb文件保存路径及文件名
session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
print
'End of programm'