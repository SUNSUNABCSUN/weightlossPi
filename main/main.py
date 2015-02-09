import cv2
import numpy as np

#structing element with 3*3 elements
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
code='0'
img = cv2.imread("1.jpg",0)	#change grey
res = cv2.resize(img,None,fx=.3,fy=.3,interpolation=cv2.INTER_CUBIC)
# res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)  #change grey, however 'img = cv2.imread("2.jpg",0)'has changed
# cv2.namedWindow("orignial image")
# # cv2.imshow("orignial image",res)

eroded = cv2.erode(res,kernel)
# cv2.namedWindow("eroded image")
# # cv2.imshow("eroded image",eroded)
dilated = cv2.dilate(res,kernel)
# cv2.namedWindow("dilated image")
# cv2.imshow("dilated image",dilated)

result = cv2.absdiff(dilated,eroded)
result = cv2.bitwise_not(result)
#cv2.namedWindow("diff result")
#cv2.imshow("diff result",result)

ret,img2=cv2.threshold(result,127,255,cv2.THRESH_BINARY)
result = cv2.bitwise_not(result)
cv2.namedWindow("img2_1")
cv2.imshow("img2_1",img2)


img2 = cv2.erode(img2, None, iterations = 4)
img2 = cv2.dilate(img2, None, iterations = 4)
img2 = cv2.bitwise_not(img2)
#cv2.namedWindow("img2_4")
#cv2.imshow("img2_4",img2)

(cnts, _) = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.cv.BoxPoints(rect))
print box

# draw a bounding box arounded the detected barcode and display the
# image
# cv2.drawContours(res, [box], -1, (0, 255, 0), 3)
#cv2.namedWindow("image")
# cv2.imshow("image", res)

pts1 = np.float32([box[2],box[3],box[1],box[0]])
pts2 = np.float32([[0,0],[500,0],[0,300],[500,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(res,M,(500,300))

#cv2.bitwise_not(dst)

cv2.namedWindow("final")
cv2.imshow("final",dst)
#GaussianBlur
dst = cv2.GaussianBlur(dst,(5,5),1.5)
# binary
ret,dst=cv2.threshold(dst,100,255,cv2.THRESH_BINARY)
cv2.namedWindow("Final_2")
cv2.imshow("Final_2",dst)

#gain the shap of the cutted pic

m, n = dst.shape[:2]
bar_y = np.zeros((500,255))
bar_num = np.zeros((500,255))
bar_int = np.zeros(500,dtype=int)   
binary_bar = np.zeros(500,dtype=int)
l=0
for i in range(1,m):
    k = 1
    l = l+1
    for j in range(1,n-1):
        if dst[i,j]!=dst[i,j+1]:
            #bar_x(l,k) = i
            bar_y[l,k]=j
            k = k+1
        if k>61:
            l = l-1
            break
    if k<61:
        l = l-1

print bar_y


m, n = bar_y.shape[:2]

if m <= 1:
    code = '0'
    print(1,'GameOver~\n')
    

for i in range(1,m):           
    for j in range(1,n-1):
        bar_num[i,j] = bar_y[i,j+1] - bar_y[i,j]
        if bar_num[i,j]<0:
            bar_num[i,j] = 0
        
    

bar_sum = sum(bar_num)/m   
k = 0
for i in range(1,59):   
    k = k + bar_sum[i]

k = k/95   
for i in range(1,59): 
    bar_int[i] = round(bar_sum[i]/k)

k = 1
for i in range(1,59):  
    if i%2 == 1:
        for j in range(1,bar_int[i]):  
            binary_bar[k] = 1
            k = k+1
        
    else:
        for j in range(1,bar_int[i]):  
            binary_bar[k] = 0
            k = k+1
print (binary_bar)       
    
#########################
#start to change the binary codes in to bar codes
#
check_left = np.int0([[13,25,19,61,35,49,47,59,55,11],[39,51,27,33,29,57, 5,17, 9,23]])
check_right = np.int0([114,102,108,66,92,78,80,68,72,116])
first_num = np.int0([31,20,18,17,12,6,3,10,9,5])
bar_left = np.zeros(7,dtype=int)
bar_right = np.zeros(7,dtype=int)
if ((binary_bar[1] and ~binary_bar[2] and binary_bar[3]) and (~binary_bar[46] and binary_bar[47] and ~binary_bar[48] and binary_bar[49] and ~binary_bar[50]) and (binary_bar[95] and ~binary_bar[94] and binary_bar[93])):
    l = 1
    #change the left binary numbers into decimal numbers
    for i in range(1,6):
        bar_left[l] = 0
        for k in range(1,7):
            bar_left[l] = bar_left[l]+binary_bar[7*(i-1)+k+3]*(2^(7-k))
        l = l+1
    l = 1
    #change the right binary numbers into decimal numbers
    for i in range(1,6):
        bar_right[l] = 0
        for k in range(1,7):
            bar_right[l] = bar_right[l]+binary_bar[7*(i+6)+k+1]*(2^(7-k))
            k = k-1
        l = l+1

num_bar = ''
num_first = 0
first = 2
#check the bar codes from the left bar dictionary
for i in range(1,6):
    for j in range(0,1):
        for k in range(0,9):
            if bar_left[i]==check_left[j+1,k+1]:
                num_bar = strcat(num_bar , num2str(k));
                
                if first == 0:
                    num_first = num_first + ~j*(2^(6-i))
                elif first == 1:
                    num_first = num_first + j*(2^(6-i))
                elif first == 2:
                    first = j


#check the bar codes from the right bar dictionary
for i in range(1,6):
    for j in range(0,9):
        if bar_right[i]==check_right[j+1]:
            num_bar = strcat(num_bar , num2str(j))

#check first bar code from the first bar code dictionary
for i in range(0,9):
    if num_first==first_num[i+1]:
        num_bar = strcat(num2str(i) , num_bar)
        break

print ('the bar code is: ',num_bar)

cv2.waitKey(0)
cv2.destroyAllWindows()