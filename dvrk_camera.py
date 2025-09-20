import numpy as np
import os
import cv2
from Kinematics import GetPositionInBaseFrame, GetPositionInCameraFrame

class dvrk_camera:
    def __init__(self, K_mat, Tcr):
        self.K = K_mat # camera intrinsic parameters
        self.Tcr = Tcr # hand-eye transformation matrix
        pass
    
    def UpdateTcr(self, Tcr_new):
        self.Tcr = Tcr_new

    def GetPositionInCameraFrame(self, position_base):
        return np.matmul(self.Tcr[:3,:3],position_base) + self.Tcr[:3,-1]

    def GetPositionInCameraFrameList(self, position_base_list):
        return [self.GetPositionInCameraFrame(pos) for pos in position_base_list]

    def PixelProjection(self, position_camera):
        fx = self.K[0,0]
        fy = self.K[1,1]
        cx = self.K[0,2]
        cy = self.K[1,2]
        homo_x = position_camera[0] / position_camera[2]
        homo_y = position_camera[1] / position_camera[2]
        u = fx*homo_x + cx
        v = fy*homo_y + cy
        return (int(u),int(v))
    
    def PixelProjectionList(self, position_camera_list):
        return [self.PixelProjection(pos) for pos in position_camera_list]

    def DrawKeyPoint(self, img, pt, radius=5, color=(0, 0, 255), thickness=-1, text = None):
        img_copy = img.copy()
        output = cv2.circle(img_copy, (pt[0],pt[1]), radius, color, thickness)
        if text is not None:
            output = cv2.putText(img_copy, text, (pt[0] - 20, pt[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        return output
    
    def DrawKeyPointsList(self, img, pt_list, radius=5, color=(0,0,255), thickness=-1, text_list = None):
        output = img.copy()
        if text_list is not None:
            assert len(text_list) == len(pt_list), "len(pt_list) is not equal to len(text_list)"
            pt_text_dic = dict(zip(pt_list, text_list))
            for pt, text in pt_text_dic.items():
                output = self.DrawKeyPoint(output, pt, radius, color, thickness,text)
        else:
            for pt in pt_list:
                output = self.DrawKeyPoint(output, pt, radius, color, thickness)
        return output
    
    def BluePointsDetection(self, img, blue_lower = np.array([94, 80, 2], np.uint8), blue_upper = np.array([120, 255, 255], np.uint8), pt_topleft = (0,0), pt_downright = (1920,1080)):
        hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H,S,V = cv2.split(hsvFrame)
        lim = 255 - 50
        V[V > lim] = 255
        V[V <= lim] += 50

        final_hsv = cv2.merge((H, S, V))
        
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
        # Creating contour to track blue color
        contours, hierarchy = cv2.findContours(blue_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        contours = contours[:15]
        KeyPointPositionList = []

        x_min, y_min = pt_topleft
        x_max, y_max = pt_downright

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 30):
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    if cx >= x_min and cx <= x_max and cy >= y_min and cy<= y_max:
                        KeyPointPositionList.append((cx,cy))
                        img = cv2.circle(img, (cx, cy), 7, (255, 0, 0), -1)
                        img = cv2.putText(img, "center", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # x, y, w, h = cv2.boundingRect(contour)
                # img = cv2.rectangle(img, (x, y),
                #                         (x + w, y + h),
                #                         (255, 0, 0), 2)

                # cv2.putText(img, "Blue Colour", (x, y),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             1.0, (255, 0, 0))
        # cv2.imshow("Color Detection", img)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        return KeyPointPositionList
    
    # AX+BY+C = 0 form
    def GetEdgeProjectionCylinder(self, radius, JointAngles):
        fx = self.K[0,0]; fy = self.K[1,1]; cx = self.K[0,2]; cy = self.K[1,2]
        joint_4_base = GetPositionInBaseFrame(JointAngles, np.array([0,0,0]), 4)
        joint_3_base = GetPositionInBaseFrame(JointAngles, np.array([0,0,0]), 3)
        joint_4_cam = GetPositionInCameraFrame(self.Tcr, joint_4_base)
        joint_3_cam = GetPositionInCameraFrame(self.Tcr, joint_3_base)
        shaft_axis_cam = (joint_3_cam-joint_4_cam) / np.linalg.norm(joint_3_cam - joint_4_cam)
        a, b, c = shaft_axis_cam
        x0, y0, z0 = joint_4_cam
        nu = a*x0 + b*y0 + c*z0
        C = x0*x0 + y0*y0 + z0*z0 - nu*nu - radius*radius
        alpha = c*y0-b*z0
        beta = a*z0-c*x0
        kappa = b*x0-a*y0
        edge_down_raw = [radius*(x0-a*nu)/np.sqrt(C)-alpha, radius*(y0-b*nu)/np.sqrt(C)-beta, radius*(z0-c*nu)/np.sqrt(C)-kappa]
        edge_up_raw = [radius*(x0-a*nu)/np.sqrt(C)+alpha, radius*(y0-b*nu)/np.sqrt(C)+beta, radius*(z0-c*nu)/np.sqrt(C)+kappa]
        edge_down = [edge_down_raw[0]/fx, edge_down_raw[1]/fy, edge_down_raw[2]-edge_down_raw[0]/fx*cx - edge_down_raw[1]/fy*cy]
        edge_up = [edge_up_raw[0]/fx, edge_up_raw[1]/fy, edge_up_raw[2]-edge_up_raw[0]/fx*cx - edge_up_raw[1]/fy*cy]
        return np.array(edge_down), np.array(edge_up)

    def DrawLines(self, img, line_list, color=(0,255,0), thickness=2):
        n_lines = len(line_list)
        assert n_lines > 0, "no lines to draw"
        output = img.copy()
        width = img.shape[1]
        height = img.shape[0]
        for line in line_list:
            A, B, C = line
            if B != 0:
                pt1 = (int(0), int(-C/B))
                pt2 = (width, int((-C-A*width)/B))
                output = cv2.line(output, pt1, pt2, color, thickness)
            else:
                pt1 = (int(-C/A), int(0))
                pt2 = (int(-C/A), height)
                output = cv2.line(output, pt1, pt2, color, thickness)
        return output

    # [[pt1,pt2], [pt2,pt3] ... ]
    def DrawToolSkeleton(self, img, line_list, color=(0,255,0), thickness=2):
        output = img.copy()
        for line in line_list:
            pt1, pt2 = line
            output = cv2.line(output, pt1, pt2, color, thickness)
        return output
    
    def DrawKeyPointsAssociation(self, img, measurement_list, reference_list, MatchedKeys, color=(0,255,0), thickness=2):
        output = img.copy()
        index = 0
        for measured_pt in measurement_list:
            if MatchedKeys[index] == None:
                index += 1
                continue
            else:
                key = MatchedKeys[index]
                ref_pt = reference_list[key]
                output = cv2.arrowedLine(output, measured_pt, ref_pt, color, thickness) 
                index += 1
        return output
    
#     def CannyEdgeDetection(self, img, t_lower=50, t_upper=150):
#         img_copy = img.copy()
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
#         blurred_image = cv2.GaussianBlur(gray_img, (5, 5), 1.4)
#         blurred_image = clahe.apply(blurred_image)
#         edges = cv2.Canny(blurred_image, t_lower, t_upper)
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, 40, 10)
#         if lines is not None:
#             for i in range(0, len(lines)):
#                 rho = lines[i][0][0]
#                 theta = lines[i][0][1]
#                 a = np.cos(theta)
#                 b = np.sin(theta)
#                 x0 = a * rho
#                 y0 = b * rho
#                 pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#                 pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#                 cv2.line(img_copy, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
#         cv2.imshow('lines', img_copy)
#         cv2.waitKey(0)
#         return edges

# def change_brightness(img, value=30):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     v = cv2.add(v,value)
#     v[v > 255] = 255
#     v[v < 0] = 0
#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img