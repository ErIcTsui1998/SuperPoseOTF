import numpy as np

def PixelProjection(position_camera, K):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    homo_x = position_camera[0] / position_camera[2]
    homo_y = position_camera[1] / position_camera[2]
    u = fx*homo_x + cx
    v = fy*homo_y + cy
    return (int(u),int(v))

# Is pixel (u,v) above line AX+BY+C=0 or not
def IsPointAboveLine(pt_pixel, line_equation):
    u, v = pt_pixel
    A,B,C = line_equation
    if A*u + B*v + C >= 0:
        return True
    else:
        return False

def IsPointAbovelineList(pt_pixel_list, line_euqaiton):
    n_total = len(pt_pixel_list)
    output = []
    for i in range(n_total):
        result = IsPointAboveLine(pt_pixel_list[i], line_euqaiton)
        output.append(result)
    return output

# Find the line equation given two points (u1,v1) and (u2,v2) in the form of AX+BY+C = 0
def GetLineEquationFromPointSet(p1, p2):
    u1, v1 = p1
    u2, v2 = p2
    if u1 == u2:
        if v1 == v2:
            raise Exception("p1 p2 coincide. The line reduces to a point")
        else: 
            return np.array([1,0.0,-u1])
    else:
        k = (v1-v2) / (u1-u2)
        b = v1 - k*u1
        return np.array([k,-1,b])

# Find the equation for a list of lines
def GetListOfLineEquationFromPointSet(pt_set_list):
    output = []
    for pt_set in pt_set_list:
        pt1, pt2 = pt_set
        line_equ = GetLineEquationFromPointSet(pt1, pt2)
        output.append(line_equ)
    return output

# pt_pixel = (u,v), line equation: AX+BY+C=0
def GetPointToLineDistance(pt_pixel, line_equation):
    u, v = pt_pixel
    A,B,C = line_equation
    return abs(A*u+B*v+C) / np.sqrt(A*A+B*B)

def GetPointToLineDistanceList(pt_pixel_list, line_equation):
    n_total = len(pt_pixel_list)
    output = []
    for i in range(n_total):
        distance = GetPointToLineDistance(pt_pixel_list[i], line_equation)
        output.append(distance)
    return output

def GetROIFromKinematics(JointPixels, margin_x=50, margin_y=30):
    x = [pixel[0] for pixel in JointPixels]
    y = [pixel[1] for pixel in JointPixels]
    return (min(x)-margin_x, min(y)-margin_y), (max(x)+margin_x, max(y)+margin_y)

def IsPointVisible(width, height, pt):
    u, v = pt
    if u<= width and u>= 0 and v<= height and v >=0:
        return True
    else:
        return False

def IsPointVisibleList(width, height, pt_list):
    output = []
    for pt in pt_list:
        result = IsPointVisible(width, height, pt)
        output.append(result)
    return output

# from AU+BY+C = 0 form to Hesse form (rho, theta)
def LineABCToHesseForm(line_equation):
    A, B, C = line_equation
    if C < 0:
        C = -C
        A = -A
        B = -B
    sqr_sum = np.sqrt(A*A + B*B)
    A = A/sqr_sum
    B = B/sqr_sum
    C = C/sqr_sum
    return (C, np.arctan2(B,A))

# Decide dominant sides 
# KeyPointsName = ["rf","rb","rr","rl","pf","pb","pr","pl","ef","eb"]
def VisibilityEvaluation(AboveCentralList, CentralDistanceList, SideDistanceList, NameList):
    # Step 1, use "rf", "rb", "rr", "rl" to determine whether it is or fb or lr facing towards the camera
    num_keypoints = len(NameList)
    assert len(NameList) == len(AboveCentralList), "name list mismatches above central list"
    assert len(NameList) == len(CentralDistanceList), "name list matches central distance list"
    AboveCentralDic = dict(zip(NameList, AboveCentralList))
    CentralDistanceDic = dict(zip(NameList, CentralDistanceList))
    SideDistanceDic = dict(zip(NameList, SideDistanceList))
    count_front_above = 0
    count_back_above = 0
    count_left_above = 0
    count_right_above = 0
    for key, value in AboveCentralDic.items():
        if key[-1] == "f" and value:
            count_front_above += 1
        elif key[-1] == "b" and value:
            count_back_above += 1
        elif key[-1] == "l" and value:
            count_left_above += 1
        elif key[-1] == "r" and value:
            count_right_above += 1
    print(f"left above = {count_left_above} \n right above = {count_right_above} \n front above = {count_front_above} \n back above = {count_back_above}")
    
    # # Ideally, the difference between back and front, left and right, should be unequivocal 
    # if count_back_above == count_front_above:
    #     print("num of back = num of front, please monitor")
    #     input("")
    # if count_left_above == count_right_above:
    #     print("num of left == num of right, please monitor")
    #     input("")

    fb_lr_ratio = (CentralDistanceDic["rf"] * CentralDistanceDic["rb"])/(CentralDistanceDic["rl"] * CentralDistanceDic["rf"])
    fb_central_side_ratio = (CentralDistanceDic["rf"] / SideDistanceDic["rf"]) + (CentralDistanceDic["rb"] / SideDistanceDic["rb"]) # the higher the ratio is, the more likely they fall on the edges
    lr_central_side_ratio = (CentralDistanceDic["rl"] / SideDistanceDic["rl"]) + (CentralDistanceDic["rr"] / SideDistanceDic["rr"]) 

    # print(f"fb cs ratio = {fb_central_side_ratio}")
    # print(f"lr cs ratio = {lr_central_side_ratio}")
    
    # case: fb dominant, front towards the camera
    threshold1 = 100
    Initscore = list(np.zeros(num_keypoints))
    ScoreList = []
    # Case 1: Single side dominantly visible
    if fb_central_side_ratio >= threshold1 or lr_central_side_ratio >= threshold1:
        # case 1.1 fb on edges 
        if fb_central_side_ratio > lr_central_side_ratio:
            # case 1.1.1 right up
            if count_front_above > count_back_above:
                ScoreList = [1.0 if name[-1]=="r" else 0.0 for name in NameList]
                print("Right side dominant")
            # case 1.1.2 left up
            elif count_back_above > count_front_above:
                ScoreList = [1.0 if name[-1]=="l" else 0.0 for name in NameList]
                print("Left side dominant")
            else:
                KeyError("num front = num back, please monitor")
        # case 1.2 lr on edges
        else:
            # case 1.2.1 front up
            if count_left_above > count_right_above:
                ScoreList = [1.0 if name[-1]=="f" else 0.0 for name in NameList]
                print("Front side dominant")
            # case 1.2.2 back up
            elif count_right_above > count_left_above:
                ScoreList = [1.0 if name[-1]=="b" else 0.0 for name in NameList]
                print("Back side dominant")  
            else:
                KeyError("num left = num right, please monitor")  
    
    # Case 2: Double sides equally visible
    else:
        # case 2.1 fb closer to edges
        if fb_central_side_ratio > lr_central_side_ratio:
            # case 2.1.1 right slightly up
            if count_front_above > count_back_above:
                # case 2.1.1.1 right and back
                if count_right_above > count_left_above:
                    # ScoreList = [1.0 if name[-1]=="r" else 0.0 for name in NameList]
                    ScoreList = [SideDistanceDic[name] / CentralDistanceDic[name] if name[-1]=="r" or name[-1]=="b" else 0.0 for name in NameList]
                    print("Right side and Back side facing towards the camera")
                # case 2.1.1.2 right and front
                elif count_left_above > count_right_above:
                    ScoreList = [SideDistanceDic[name] / CentralDistanceDic[name] if name[-1]=="r" or name[-1]=="f" else 0.0 for name in NameList]
                    print("Right side and Front side facing towards the camera")
                else:
                    KeyError("num left = num right, please monitor") 
            # case 2.1.2 left slightly up
            elif count_back_above > count_front_above:
                # case 2.1.2.1 left and front
                if count_left_above > count_right_above:
                    print("Left side and Front side facing towards the camera")
                    ScoreList = [SideDistanceDic[name] / CentralDistanceDic[name] if name[-1]=="l" or name[-1]=="f" else 0.0 for name in NameList]
                # case 2.1.2.2 left and back
                elif count_right_above > count_left_above:
                    ScoreList = [SideDistanceDic[name] / CentralDistanceDic[name] if name[-1]=="l" or name[-1]=="b" else 0.0 for name in NameList]
                    print("Left side and Back side facing towards the camera")
                else:
                    KeyError("num left = num right, please monitor") 
            else:
                KeyError("num front = num back, please monitor") 
        # case 2.2 lr closer to edges
        elif lr_central_side_ratio > fb_central_side_ratio:
            # case 2.2.1 back slightly up
            if count_right_above > count_left_above:
                # case 2.2.1.1 back and left
                if count_back_above > count_front_above:
                    ScoreList = [SideDistanceDic[name] / CentralDistanceDic[name] if name[-1]=="b" or name[-1]=="l" else 0.0 for name in NameList]
                    print("Back side and left side facing towards the camera")
                # case 2.2.1.2 back and right
                elif count_front_above > count_back_above:
                    ScoreList = [SideDistanceDic[name] / CentralDistanceDic[name] if name[-1]=="b" or name[-1]=="r" else 0.0 for name in NameList]
                    print("Back side and right side facing towards the camera")
                else:
                    KeyError("num back = num_front, please monitor")
            # case 2.2.2 front slightly up
            elif count_left_above > count_right_above:
                # case 2.2.2.1 front and right
                if count_front_above > count_back_above:
                    ScoreList = [SideDistanceDic[name] / CentralDistanceDic[name] if name[-1]=="f" or name[-1]=="r" else 0.0 for name in NameList]
                    print("Front side and right side facing towards the camera")
                # case 2.2.2.2 front and left
                elif count_back_above > count_front_above:
                    ScoreList = [SideDistanceDic[name] / CentralDistanceDic[name] if name[-1]=="f" or name[-1]=="l" else 0.0 for name in NameList]
                    print("Front side and left side facing towards the camera")
                else:
                    KeyError("num back = num front, please monitor")
            else:
                KeyError("num right = num left, please monitor")
        else:
            print("fb ratio = lr ratio, please monitor")
            input(" ")

    return ScoreList