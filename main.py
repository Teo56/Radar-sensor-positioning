import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve

time = []
s1_distance = []
s2_distance = []
s3_distance = []
s4_distance = []
distance = []
distance_total = []

with open('file_1.csv', 'r', newline='') as file:
    #reader = csv.reader(file)
    reader = csv.reader(x.replace('\0', '') for x in file) # Since the file contains null characters, they are substituted using space ' '
    next(reader, None) # Skips the header of the file
    for row in reader:
        time.append(float(row[0])/1000)
        s1_distance.append(float(row[1]))
        s2_distance.append(float(row[2]))
        s3_distance.append(float(row[3]))
        s4_distance.append(float(row[4]))
        distance.append(float(row[1]))
        distance.append(float(row[2]))
        distance.append(float(row[3]))
        distance.append(float(row[4]))
        distance_total.append(distance)
        distance = []

plt.plot(time, s1_distance)
plt.plot(time, s2_distance)
plt.plot(time, s3_distance)
plt.plot(time, s4_distance)

plt.title("sensors vs time - original")
plt.show()


# CREATE DATA FRAME
# ----------------
data = {'time': time, 's1_distance': s1_distance, 's2_distance': s2_distance, 's3_distance': s3_distance, 's4_distance': s4_distance}

df = pd.DataFrame(data, columns=['time', 's1_distance', 's2_distance', 's3_distance', 's4_distance'])
print(df)

# NB must use 0:1 otherwise it does not work
X = df.iloc[:, 0:1].values #take the values in the first column (time) - save them in array X
y = df.iloc[:, 1].values #take the values in the second column (s1_distance) - save them in array y


#df.plot( x= 'time', y = 's1_distance')
# TAKE MOVING AVERAGE
# here can select the number of data to use for the moving average
avg_number = 5
df.rolling(window = avg_number).mean().plot( x= 'time', y = 's1_distance')
df.rolling(window = avg_number).mean().plot( x= 'time', y = 's2_distance')
df.rolling(window = avg_number).mean().plot( x= 'time', y = 's3_distance')
df.rolling(window = avg_number).mean().plot( x= 'time', y = 's4_distance')
plt.show()

# add the new column of the moving avg
df['moving_avg_s1'] = df['s1_distance'].rolling(window=avg_number).mean()
df['moving_avg_s2'] = df['s2_distance'].rolling(window=avg_number).mean()
df['moving_avg_s3'] = df['s3_distance'].rolling(window=avg_number).mean()
df['moving_avg_s4'] = df['s4_distance'].rolling(window=avg_number).mean()

print ("\n\nNEW DATA")
print(df)

# Convert into a list
s1_avg_with_nan = df['moving_avg_s1'].tolist()
s2_avg_with_nan = df['moving_avg_s2'].tolist()
s3_avg_with_nan = df['moving_avg_s3'].tolist()
s4_avg_with_nan = df['moving_avg_s4'].tolist()

print(s1_avg_with_nan)
print("-----------------\n")


# TO REMOVE NAN VALUES FROM THE LISTS:
s1_avg = [x for x in s1_avg_with_nan if math.isnan(x) == False]
s2_avg = [x for x in s2_avg_with_nan if math.isnan(x) == False]
s3_avg = [x for x in s3_avg_with_nan if math.isnan(x) == False]
s4_avg = [x for x in s4_avg_with_nan if math.isnan(x) == False]

print(s1_avg)

# needed because nan values have been removed
for i in range(avg_number-1):
    time.pop(0)
    s1_distance.pop(0)
    s2_distance.pop(0)
    s3_distance.pop(0)
    s4_distance.pop(0)

plt.plot(time, s1_avg)
plt.plot(time, s1_distance)

plt.plot(time, s2_avg)
plt.plot(time, s2_distance)

plt.plot(time, s3_avg)
plt.plot(time, s3_distance)

plt.plot(time, s4_avg)
plt.plot(time, s4_distance)

plt.show()


# ------------------------------------------
# Find moments of STILLNESS (NO MOVEMENT)

still_indexes = []
sensor_index = []
walking_indexes = []
indicator_1 = 0
indicator_2 = 0
indicator_3 = 0
indicator_4 = 0

still_indicator = 1
i = 0

min_walking_distance = 10
min_still = 5

# INDEXES PERSON NOT MOVING
# -------------------------
while i < (len(s1_avg)-1):
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    sum_4 = 0
    for q in range(min_still):
        if (i+q)<len(s1_avg):
            sum_1 = sum_1 + s1_avg[i+q]
            sum_2 = sum_2 + s2_avg[i+q]
            sum_3 = sum_3 + s3_avg[i+q]
            sum_4 = sum_4 + s4_avg[i+q]

    # Find avg for each sequence of distances
    avg_1 = sum_1/min_still
    avg_2 = sum_2/min_still
    avg_3 = sum_3/min_still
    avg_4 = sum_4/min_still

    # PERSON NOT MOVING
    for q in range(min_still):
        if abs(s1_avg[i+q] - avg_1) < 8 and abs(s2_avg[i+q] - avg_2) < 8 and abs(s3_avg[i+q] - avg_3) < 8 and abs(s4_avg[i+q] - avg_4) < 8:
            still_indicator = 0
        else:
            still_indicator=1
            break

    if still_indicator==0:
        for y in range(min_still):
            still_indexes.append(i+y)
            #print("--", round((time[i] - 1622132000), 3)) # to check with the graph
        i=i+(min_still-1)

    i = i + 1
    still_indicator = 1


# INDEXES PERSON MOVING IN STRAIGHT LINE
# --------------------------------------
i = 0
move_index = []
walking_test_1 = []
walking_test_2 = []
walking_test_3 = []
walking_test_4 = []
indicator = 0
mse_array = []
variation_array = []

# function to calculate the mse of the elements of an array
def mse(a):
    b = []
    mean = np.mean(a)
    for i in range(len(a)):
        b.append(abs(a[i]-mean))
    mse = np.mean(b)
    return mse

# FROM HERE, CHECK MOTION ON STRAIGHT LINE
# -------------------------
while i + min_walking_distance < (len(s1_avg) - 1):
    print("\n1\n")

#if i not in still_indexes:
    for q in range(min_walking_distance):
        if i+q not in still_indexes:
            walking_test_1.append(s1_avg[i + q + 1] - s1_avg[i + q])
            walking_test_2.append(s2_avg[i + q + 1] - s2_avg[i + q])
            walking_test_3.append(s3_avg[i + q + 1] - s3_avg[i + q])
            walking_test_4.append(s4_avg[i + q + 1] - s4_avg[i + q])
        else:
            indicator = 1
   # print("\n\indicator: ")
   # print(indicator)
    print("\n2\n")
    if indicator == 0:
        print("\n2.5\n")

        if mse(walking_test_1)<3:
            #print("yessss 1")
            mse_array.append(mse(walking_test_1))
        else:
            mse_array.append(0)
        if mse(walking_test_2)<3:
            #print("yessss 2")
            mse_array.append(mse(walking_test_2))
        else:
            mse_array.append(0)
        if mse(walking_test_3)<3:
            #print("yessss 3")
            mse_array.append(mse(walking_test_3))
        else:
            mse_array.append(0)
        if mse(walking_test_4)<3:
            #print("yessss 4")
            mse_array.append(mse(walking_test_4))
        else:
            mse_array.append(0)

        variation_array.append(np.absolute(np.mean(walking_test_1)))
        variation_array.append(np.absolute(np.mean(walking_test_2)))
        variation_array.append(np.absolute(np.mean(walking_test_3)))
        variation_array.append(np.absolute(np.mean(walking_test_4)))
        print("\n3\n")
        print(variation_array)

        while len(move_index) == 0:
            sensor_max_var = variation_array.index(max(variation_array))
            print("sensor_max_var :")
            print(sensor_max_var)
            print("mse_array :")
            print(mse_array)
            # exit()

            if mse_array[sensor_max_var] != 0:
                move_index.append(sensor_max_var)
                print("\n4\n")
                #exit()
                for t in range(min_walking_distance):
                    move_index.append(i + t)
                i = i + min_walking_distance - 1
            else:
                variation_array[sensor_max_var] = 0
            print("hello")
            #exit()

        walking_indexes.append(move_index)
        move_index = []
        variation_array = []
        mse_array = []
        print("\n5\n")


    walking_test_1 = []
    walking_test_2 = []
    walking_test_3 = []
    walking_test_4 = []
    indicator = 0
    indicator_1 = 0
    indicator_2 = 0
    indicator_3 = 0
    indicator_4 = 0

    i = i + 1

print("\nmse array: \n")
print(mse_array)
print("\nvariation array: \n")
print(variation_array)


print("still_indexes: ", still_indexes)
print("walking_indexes: ", walking_indexes)
# walking_index:
# sensor - index - index - index - .....
# sensor - index - index - index - .....
# sensor - index - index - index - .....
# sensor - index - index - index - .....


plt.scatter(time, s1_avg)
plt.scatter(time, s2_avg)
plt.scatter(time, s3_avg)
plt.scatter(time, s4_avg)


# STILL
for t in still_indexes:
    plt.plot(time[t], s1_avg[t], marker='x', markerfacecolor='blue', markersize=8)
    plt.plot(time[t], s2_avg[t], marker='x', markerfacecolor='blue', markersize=8)
    plt.plot(time[t], s3_avg[t], marker='x', markerfacecolor='blue', markersize=8)
    plt.plot(time[t], s4_avg[t], marker='x', markerfacecolor='blue', markersize=8)

# MOVING
# NB some points will overlap --> plotted more than once !!!! because the calculation is not shifted when a straight line sequence is found
for t in walking_indexes:
    sensor = t[0]
    if sensor==0:
        for r in range(min_walking_distance):
            plt.plot(time[t[1+r]], s1_avg[t[1+r]], marker='o', markerfacecolor='orange', markersize=8)

        """for i in range(min_walking_distance):
            if t[1]+min_walking_distance <len(s1_avg):
                plt.plot(time[t[1]+i], s1_avg[t[1]+i], marker='o', markerfacecolor='orange', markersize=8)"""
    if sensor==1:
        for r in range(min_walking_distance):
            plt.plot(time[t[1 + r]], s2_avg[t[1 + r]], marker='o', markerfacecolor='orange', markersize=8)

        """for i in range(min_walking_distance):
            if t[1]+min_walking_distance  < len(s2_avg):
                plt.plot(time[t[1]+i], s2_avg[t[1]+i], marker='o', markerfacecolor='orange', markersize=8)"""
    if sensor==2:
        for r in range(min_walking_distance):
            plt.plot(time[t[1 + r]], s3_avg[t[1 + r]], marker='o', markerfacecolor='orange', markersize=8)

        """for i in range(min_walking_distance):
            if t[1]+min_walking_distance  < len(s1_avg):
             plt.plot(time[t[1]+i], s3_avg[t[1]+i], marker='o', markerfacecolor='orange', markersize=8)"""
    if sensor==3:
        for r in range(min_walking_distance):
            plt.plot(time[t[1 + r]], s4_avg[t[1 + r]], marker='o', markerfacecolor='orange', markersize=8)

        """for i in range(min_walking_distance):
            if t[1]+min_walking_distance  < len(s1_avg):
                plt.plot(time[t[1]+i], s4_avg[t[1]+i], marker='o', markerfacecolor='orange', markersize=8)"""

plt.title("S1: blue - S2: orange - S3: green - S4: red")
plt.show()



# ----------------------------
# THREE METHODS FOR STEP ESTIMATION:

# METHOD 1: step = max variation among data --> too many errors
# Find the approximate value of the step length as the greatest variation in distance from one of the sensors

step = 0
for i in range(len(s1_avg)-1):
    if not math.isnan(s1_avg[i]):
        variation_1 = abs(s1_avg[i + 1] - s1_avg[i])
        variation_2 = abs(s2_avg[i + 1] - s2_avg[i])
        variation_3 = abs(s3_avg[i + 1] - s3_avg[i])
        variation_4 = abs(s4_avg[i + 1] - s4_avg[i])
        max_var = max(variation_1, variation_2, variation_3, variation_4)

        if max_var > step:
            step = max_var
            index = i

print("step: ", step)
print("index: ", index)

# METHOD 2: Take 10 max variations for each sensor, make the average of each and take the average of each average
step = 0
n = 10
s1_20 = []
s2_20 = []
s3_20 = []
s4_20 = []

mean_1 = 0
mean_2 = 0
mean_3 = 0
mean_4 = 0

def largest_n(a, b, n):
    for t in range(n):
        b.append(abs(a[t+1]-a[t]))

    for i in range(len(a) - (n+1)):
        if (abs(a[i+n+1]-a[i+n]) > min(b)):
            index = b.index(min(b))
            b[index] = abs(a[i+n+1]-a[i+n])
    return b

print(largest_n(s1_avg, s1_20, n))
print(largest_n(s2_avg, s2_20, n))
print(largest_n(s3_avg, s3_20, n))
print(largest_n(s4_avg, s4_20, n))

mean_1 = sum(largest_n(s1_avg, s1_20, n))/len(largest_n(s1_avg, s1_20, n))
mean_2 = sum(largest_n(s2_avg, s2_20, n))/len(largest_n(s2_avg, s2_20, n))
mean_3 = sum(largest_n(s3_avg, s3_20, n))/len(largest_n(s3_avg, s3_20, n))
mean_4 = sum(largest_n(s4_avg, s4_20, n))/len(largest_n(s4_avg, s4_20, n))

step = (mean_1+mean_2+mean_3+mean_4)/4

print ("Step with second method: ")
print(step)

# THIRD METHOD: take the average of each section where the person is moving in a straight line and then take the average of those averages
avg_vector = []
total_avg = 0
for t in walking_indexes:
    total = 0
    avg = 0
    #sensor = "s" + str(t[0]+1) +"_avg"
    sensor = t[0]
    i = 1
    while i<(len(t)-1):
        #exec("sum = sum + sensor[i + 1] - sensor[i]")
        #exec("sum = sum + s" + str(t[0]+1) +"avg[" + str(i+1)+ "] - s"  + str(t[0]+1) +"avg[" + str(i)+ "]")
        index = t[i]

        if sensor == 0:
            total = total + abs(s1_avg[index+1]-s1_avg[index])
        elif sensor == 1:
            total = total + abs(s2_avg[index+1]-s2_avg[index])
        elif sensor == 2:
            total = total + abs(s3_avg[index + 1] - s3_avg[index])
        elif sensor == 3:
            total = total + abs(s4_avg[index + 1] - s4_avg[index])

        i = i+1
    avg = total / (len(t) - 1)
    avg_vector.append(avg)
total_avg = sum(avg_vector) / len(avg_vector)

print("Step third method: ")
print(total_avg)

# ---------------------------------
# STEP FOURTH EASY METHOD
# USE STEP = 12
step = 12

# DEFINE THE ROTATION FUNCTION
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

# DEFINE THE ANGLE TO ROTATE BY

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


#---------------------------------
# FIRST METHOD: overlap first sensor
# - constant step = 12
# - all values NOT in still indexes
# - overlap first sensor
# - only one rotation

X_11 = []
X_12 = []
Y_11 = []
Y_12 = []
X_21 = []
X_22 = []
Y_21 = []
Y_22 = []
X_31 = []
X_32 = []
Y_31 = []
Y_32 = []
X_41 = []
X_42 = []
Y_41 = []
Y_42 = []

step = 12

#for t in range(len(distance_total)-1):
for t in range(len(distance_total)-1):
    if t not in still_indexes: # to try limit the error due to standing still
        for i in range(4):
            def f(x):
                # APPROX: STEP LENGTH AS OBTAINED BEFORE
                f0 = pow((x[0]), 2) + pow(x[1], 2) - pow(distance_total[t][i], 2)
                f1 = pow((step - x[0]), 2) + pow(x[1], 2) - pow(distance_total[t + 1][i], 2)

                return np.array([f0, f1])

            x0_1 = np.array([1000, 1000])
            x0_2 = np.array([-1000, -1000])
            solution_1 = fsolve(f, x0_1)
            solution_2 = fsolve(f, x0_2)
            print("\n", solution_1)
            print("\n", solution_2)

            if i == 0:

                X_11.append(solution_1[0])
                X_12.append(solution_2[0])
                Y_11.append(solution_1[1])
                Y_12.append(solution_2[1])

            if i == 1:

                X_21.append(solution_1[0])
                X_22.append(solution_2[0])
                Y_21.append(solution_1[1])
                Y_22.append(solution_2[1])

            if i == 2:

                X_31.append(solution_1[0])
                X_32.append(solution_2[0])
                Y_31.append(solution_1[1])
                Y_32.append(solution_2[1])

            if i == 3:

                X_41.append(solution_1[0])
                X_42.append(solution_2[0])
                Y_41.append(solution_1[1])
                Y_42.append(solution_2[1])

        """# DRAW THE CHANGING POSITION OF THE SENSORS (SOLUTION_1) (WITH RESPECT TO POINT (0,0) RECALCULATED EACH TIME)
        #x = np.linspace(0, 10, 1000)
        plt.plot(X_11, Y_11, marker='o', markerfacecolor='blue', markersize=4)
        plt.plot(X_21, Y_21, marker='o', markerfacecolor='blue', markersize=4)
        plt.plot(X_31, Y_31, marker='o', markerfacecolor='blue', markersize=4)
        plt.plot(X_41, Y_41, marker='o', markerfacecolor='blue', markersize=4)
        plt.plot(0, 0, marker='x', markerfacecolor='black', markersize=6)
        plt.show()"""

        """if t > 40:
            # ONLY DRAW THE LAST POSITION OF THE SENSORS (SOLUTION_1) WHEN THE PERSON MOVES
            plt.plot(X_11[t], Y_11[t], marker='o', markerfacecolor='blue', markersize=6)
            plt.plot(X_21[t], Y_21[t], marker='o', markerfacecolor='blue', markersize=6)
            plt.plot(X_31[t], Y_31[t], marker='o', markerfacecolor='blue', markersize=6)
            plt.plot(X_41[t], Y_41[t], marker='o', markerfacecolor='blue', markersize=6)
            plt.plot(0, 0, marker='x', markerfacecolor='black', markersize=6)
            plt.title(1622132062700 + t*100)
            plt.show()"""

# define the new list of coordinates shifted
X11_shift = []
X12_shift = []
Y11_shift = []
Y12_shift = []

X21_shift = []
X22_shift = []
Y21_shift = []
Y22_shift = []

X31_shift = []
X32_shift = []
Y31_shift = []
Y32_shift = []

X41_shift = []
X42_shift = []
Y41_shift = []
Y42_shift = []

x_shift = 0
y_shift = 0

# Start by moving everything to overlap the first sensor
t = 0
while t < len(X_11):
    x_shift = X_11[t] - X_11[0]
    y_shift = Y_11[t] - Y_11[0]

    X11_shift.append(X_11[0])
    Y11_shift.append(Y_11[0])

    X21_shift.append(X_21[t] - x_shift)
    Y21_shift.append(Y_21[t] - y_shift)

    X31_shift.append(X_31[t] - x_shift)
    Y31_shift.append(Y_31[t] - y_shift)

    X41_shift.append(X_41[t] - x_shift)
    Y41_shift.append(Y_41[t] - y_shift)

    t = t + 1


# ALL SENSORS - ALL MEASUREMENTS - SHIFTED (S1 ALL OVERLAP)
plt.plot(X11_shift, Y11_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_shift, Y21_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_shift, Y31_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_shift, Y41_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(0, 0, marker='x', markerfacecolor='black', markersize=6)
#plt.title(1622132062700 + t*100)
plt.title("FIRST method - shift only")
plt.savefig("1_shift.png")
plt.show()

# NOW ROTATE EVERYTHING BY A CERTAIN AMOUNT

X11_rotate = []
X12_rotate = []
Y11_rotate = []
Y12_rotate = []

X21_rotate = []
X22_rotate = []
Y21_rotate = []
Y22_rotate = []

X31_rotate = []
X32_rotate = []
Y31_rotate = []
Y32_rotate = []

X41_rotate = []
X42_rotate = []
Y41_rotate = []
Y42_rotate = []

for t in range(len(X11_shift)):
    origin = (X11_shift[0], Y11_shift[0])
    point = (X21_shift[0], Y21_shift[0])
    alpha = getAngle((X21_shift[t], Y21_shift[t]), origin, point)

    X11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[0])
    Y11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[1])

    X21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[0])
    Y21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[1])

    X31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[0])
    Y31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[1])

    X41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[0])
    Y41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[1])

plt.plot(X11_rotate, Y11_rotate, marker='o', markerfacecolor='black', markersize=6)
plt.plot(X21_rotate, Y21_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_rotate, Y31_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_rotate, Y41_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.title('FIRST method- shift and rotation')
plt.savefig("1_rotation.png")
plt.show()

print(X11_rotate)

# TRY TO PLOT THE AVERAGE POSITION
X1_avg_1 = sum(X11_rotate)/len(X11_rotate)
Y1_avg_1 = sum(Y11_rotate)/len(Y11_rotate)

X2_avg_1 = sum(X21_rotate)/len(X21_rotate)
Y2_avg_1 = sum(Y21_rotate)/len(Y21_rotate)

X3_avg_1 = sum(X31_rotate)/len(X31_rotate)
Y3_avg_1 = sum(Y31_rotate)/len(Y31_rotate)

X4_avg_1 = sum(X41_rotate)/len(X41_rotate)
Y4_avg_1 = sum(Y41_rotate)/len(Y41_rotate)

plt.plot(X1_avg_1, Y1_avg_1, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X2_avg_1, Y2_avg_1, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X3_avg_1, Y3_avg_1, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X4_avg_1, Y4_avg_1, marker='o', markerfacecolor='blue', markersize=6)

plt.annotate("S1", (X1_avg_1, Y1_avg_1))
plt.annotate("S2", (X2_avg_1, Y2_avg_1))
plt.annotate("S3", (X3_avg_1, Y3_avg_1))
plt.annotate("S4", (X4_avg_1, Y4_avg_1))

plt.title('FIRST method - average sensor location')
plt.savefig("1_avg.png")
plt.show()

#---------------------------
# NOW CALCULATE POSITION USING SECOND METHOD - CENTRE THE MIDDLE POINT ---------------------------
# - step = 12
# – all values not in still indexes
# -  shift to average
# – only one rotation

X11_shift = []
X12_shift = []
Y11_shift = []
Y12_shift = []

X21_shift = []
X22_shift = []
Y21_shift = []
Y22_shift = []

X31_shift = []
X32_shift = []
Y31_shift = []
Y32_shift = []

X41_shift = []
X42_shift = []
Y41_shift = []
Y42_shift = []

X_avg = (X_11[0] + X_21[0] + X_31[0] + X_41[0])/4
Y_avg = (Y_11[0] + Y_21[0] + Y_31[0] + Y_41[0])/4

t = 0
while t < len(X_11):
    x_shift = (X_11[t] + X_21[t] + X_31[t] + X_41[t])/4 - X_avg
    y_shift = (Y_11[t] + Y_21[t] + Y_31[t] + Y_41[t])/4 - Y_avg

    X11_shift.append(X_11[t] - x_shift)
    Y11_shift.append(Y_11[t] - y_shift)

    X21_shift.append(X_21[t] - x_shift)
    Y21_shift.append(Y_21[t] - y_shift)

    X31_shift.append(X_31[t] - x_shift)
    Y31_shift.append(Y_31[t] - y_shift)

    X41_shift.append(X_41[t] - x_shift)
    Y41_shift.append(Y_41[t] - y_shift)

    t = t + 1

plt.plot(X11_shift, Y11_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_shift, Y21_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_shift, Y31_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_shift, Y41_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.title("SECOND method - shift only")
plt.savefig("2_shift.png")
plt.show()

# NOW THE ROTATION PART

X11_rotate = []
X12_rotate = []
Y11_rotate = []
Y12_rotate = []

X21_rotate = []
X22_rotate = []
Y21_rotate = []
Y22_rotate = []

X31_rotate = []
X32_rotate = []
Y31_rotate = []
Y32_rotate = []

X41_rotate = []
X42_rotate = []
Y41_rotate = []
Y42_rotate = []

for t in range(len(X11_shift)):
    origin = (X_avg, Y_avg)
    point = (X11_shift[0], Y11_shift[0])
    alpha = getAngle((X11_shift[t], Y11_shift[t]), origin, point)

    X11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[0])
    Y11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[1])

    X21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[0])
    Y21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[1])

    X31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[0])
    Y31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[1])

    X41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[0])
    Y41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[1])

plt.plot(X11_rotate, Y11_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_rotate, Y21_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_rotate, Y31_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_rotate, Y41_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.title("SECOND method - shift and rotation")
plt.savefig("2_rotation.png")
plt.show()


X1_avg_2 = sum(X11_rotate)/len(X11_rotate)
X2_avg_2 = sum(X21_rotate)/len(X21_rotate)
X3_avg_2 = sum(X31_rotate)/len(X31_rotate)
X4_avg_2 = sum(X41_rotate)/len(X41_rotate)
Y1_avg_2 = sum(Y11_rotate)/len(Y11_rotate)
Y2_avg_2 = sum(Y21_rotate)/len(Y21_rotate)
Y3_avg_2 = sum(Y31_rotate)/len(Y31_rotate)
Y4_avg_2 = sum(Y41_rotate)/len(Y41_rotate)


plt.plot(X1_avg_2, Y1_avg_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X2_avg_2, Y2_avg_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X3_avg_2, Y3_avg_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X4_avg_2, Y4_avg_2, marker='o', markerfacecolor='blue', markersize=6)

plt.annotate("S1", (X1_avg_2, Y1_avg_2))
plt.annotate("S2", (X2_avg_2, Y2_avg_2))
plt.annotate("S3", (X3_avg_2, Y3_avg_2))
plt.annotate("S4", (X4_avg_2, Y4_avg_2))

plt.title("SECOND method - Average sensor locations")
plt.savefig("2_avg.png")
plt.show()


# THIRD METHOD, SHIFT THE INTERSECTION, AND ROTATION BY AN AMOUNT TO ACHIEVE BEST POSSIBLE AVERAGE PRECISION
# TAKE THE ANGLE BETWEEN SENSOR LOCATION AND BASE
# - step = 12
# – all values not in still indexes
# - shift to intersection
# – double rotation

X11_shift = []
X12_shift = []
Y11_shift = []
Y12_shift = []

X21_shift = []
X22_shift = []
Y21_shift = []
Y22_shift = []

X31_shift = []
X32_shift = []
Y31_shift = []
Y32_shift = []

X41_shift = []
X42_shift = []
Y41_shift = []
Y42_shift = []

def get_m_q (x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    q = y1-m*x1
    return [m, q]

def intersection (m1, q1, m2, q2):
    X_intersection = (q2-q1)/(m1-m2)
    Y_intersection = m1*X_intersection+q1
    return [X_intersection, Y_intersection]

# find intersection within the area... still to do

m1 = get_m_q(X_11[0],Y_11[0], X_31[0], Y_31[0])[0]
q1 = get_m_q(X_11[0],Y_11[0], X_31[0], Y_31[0])[1]
m2 = get_m_q(X_21[0],Y_21[0], X_41[0], Y_41[0])[0]
q2 = get_m_q(X_21[0],Y_21[0], X_41[0], Y_41[0])[1]

X_avg = intersection(m1, q1, m2, q2)[0]
Y_avg = intersection(m1, q1, m2, q2)[1]

t = 0
while t < len(X_11):
    m1 = get_m_q(X_11[t], Y_11[t], X_31[t], Y_31[t])[0]
    q1 = get_m_q(X_11[t], Y_11[t], X_31[t], Y_31[t])[1]
    m2 = get_m_q(X_21[t], Y_21[t], X_41[t], Y_41[t])[0]
    q2 = get_m_q(X_21[t], Y_21[t], X_41[t], Y_41[t])[1]

    X_centre = intersection(m1, q1, m2, q2)[0]
    Y_centre = intersection(m1, q1, m2, q2)[1]

    x_shift = X_centre - X_avg
    y_shift = Y_centre - Y_avg

    X11_shift.append(X_11[t] - x_shift)
    Y11_shift.append(Y_11[t] - y_shift)

    X21_shift.append(X_21[t] - x_shift)
    Y21_shift.append(Y_21[t] - y_shift)

    X31_shift.append(X_31[t] - x_shift)
    Y31_shift.append(Y_31[t] - y_shift)

    X41_shift.append(X_41[t] - x_shift)
    Y41_shift.append(Y_41[t] - y_shift)

    t = t + 1

plt.plot(X11_shift, Y11_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_shift, Y21_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_shift, Y31_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_shift, Y41_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X11_shift, Y11_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_avg, Y_avg, marker='x', markerfacecolor='red', markersize=8)
plt.title("THIRD method - shift only")
plt.savefig("3_shift.png")
plt.show()

# The 4 sensors at each step, after shift
"""for i in range(len(X11_shift)):
    plt.plot(X11_shift[i], Y11_shift[i], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X21_shift[i], Y21_shift[i], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X31_shift[i], Y31_shift[i], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X41_shift[i], Y41_shift[i], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X11_shift[i], Y11_shift[i], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X_avg, Y_avg, marker='x', markerfacecolor='red', markersize=8)
    plt.title("third method - shift only")
    plt.show()"""

# now third method rotation
X11_rotate = []
X12_rotate = []
Y11_rotate = []
Y12_rotate = []

X21_rotate = []
X22_rotate = []
Y21_rotate = []
Y22_rotate = []

X31_rotate = []
X32_rotate = []
Y31_rotate = []
Y32_rotate = []

X41_rotate = []
X42_rotate = []
Y41_rotate = []
Y42_rotate = []


for t in range(len(X11_shift)):
    origin = (X_avg, Y_avg)
    point = (X11_shift[0], Y11_shift[0])
    alpha = getAngle((X11_shift[t], Y11_shift[t]), origin, point)

    X11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[0])
    Y11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[1])

    X21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[0])
    Y21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[1])

    X31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[0])
    Y31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[1])

    X41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[0])
    Y41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[1])

# second rotation still to be done .....
plt.plot(X11_rotate, Y11_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_rotate, Y21_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_rotate, Y31_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_rotate, Y41_rotate, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_avg, Y_avg, marker='x', markerfacecolor='red', markersize=8)
plt.title("THIRD method - shift and first rotation")
plt.savefig("3_rotation1.png")
plt.show()

# SECOND rotation

X11_rotate2 = []
X12_rotate2 = []
Y11_rotate2 = []
Y12_rotate2 = []

X21_rotate2 = []
X22_rotate2 = []
Y21_rotate2 = []
Y22_rotate2 = []

X31_rotate2 = []
X32_rotate2 = []
Y31_rotate2 = []
Y32_rotate2 = []

X41_rotate2 = []
X42_rotate2 = []
Y41_rotate2 = []
Y42_rotate2 = []

for t in range(len(X11_rotate)):
    origin = (X_avg, Y_avg)
    point = (X21_rotate[0], Y21_rotate[0])
    # divide by 2 to get average position and same angle to all sensors
    # check that the measured angle is the proper one
    alpha = getAngle((X21_rotate[t], Y21_rotate[t]), origin, point)
    if alpha > 90:
        alpha = 360-(0.5*alpha)

    X11_rotate2.append(rotate(origin, (X11_rotate[t], Y11_rotate[t]), math.radians(alpha))[0])
    Y11_rotate2.append(rotate(origin, (X11_rotate[t], Y11_rotate[t]), math.radians(alpha))[1])

    X21_rotate2.append(rotate(origin, (X21_rotate[t], Y21_rotate[t]), math.radians(alpha))[0])
    Y21_rotate2.append(rotate(origin, (X21_rotate[t], Y21_rotate[t]), math.radians(alpha))[1])

    X31_rotate2.append(rotate(origin, (X31_rotate[t], Y31_rotate[t]), math.radians(alpha))[0])
    Y31_rotate2.append(rotate(origin, (X31_rotate[t], Y31_rotate[t]), math.radians(alpha))[1])

    X41_rotate2.append(rotate(origin, (X41_rotate[t], Y41_rotate[t]), math.radians(alpha))[0])
    Y41_rotate2.append(rotate(origin, (X41_rotate[t], Y41_rotate[t]), math.radians(alpha))[1])

plt.plot(X11_rotate2, Y11_rotate2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_rotate2, Y21_rotate2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_rotate2, Y31_rotate2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_rotate2, Y41_rotate2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_avg, Y_avg, marker='x', markerfacecolor='red', markersize=8)
plt.title("THIRD method - shift and double rotation")
plt.savefig("3_rotation2.png")
plt.show()

X1_avg_3 = sum(X11_rotate2)/len(X11_rotate2)
X2_avg_3 = sum(X21_rotate2)/len(X21_rotate2)
X3_avg_3 = sum(X31_rotate2)/len(X31_rotate2)
X4_avg_3 = sum(X41_rotate2)/len(X41_rotate2)
Y1_avg_3 = sum(Y11_rotate2)/len(Y11_rotate2)
Y2_avg_3 = sum(Y21_rotate2)/len(Y21_rotate2)
Y3_avg_3 = sum(Y31_rotate2)/len(Y31_rotate2)
Y4_avg_3 = sum(Y41_rotate2)/len(Y41_rotate2)

"""X1_avg_3 = sum(X11_rotate)/len(X11_rotate)
X2_avg_3 = sum(X21_rotate)/len(X21_rotate)
X3_avg_3 = sum(X31_rotate)/len(X31_rotate)
X4_avg_3 = sum(X41_rotate)/len(X41_rotate)
Y1_avg_3 = sum(Y11_rotate)/len(Y11_rotate)
Y2_avg_3 = sum(Y21_rotate)/len(Y21_rotate)
Y3_avg_3 = sum(Y31_rotate)/len(Y31_rotate)
Y4_avg_3 = sum(Y41_rotate)/len(Y41_rotate)"""


plt.plot(X1_avg_3, Y1_avg_3, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X2_avg_3, Y2_avg_3, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X3_avg_3, Y3_avg_3, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X4_avg_3, Y4_avg_3, marker='o', markerfacecolor='blue', markersize=6)

plt.annotate("S1", (X1_avg_3, Y1_avg_3))
plt.annotate("S2", (X2_avg_3, Y2_avg_3))
plt.annotate("S3", (X3_avg_3, Y3_avg_3))
plt.annotate("S4", (X4_avg_3, Y4_avg_3))

plt.title("THIRD method - average sensor location")
plt.savefig("3_avg.png")
plt.show()


# FOURTH METHOD: SELECTION METHOD
# - use only walking indexes (straight line condition)
# - step = displacement of the person (variation of distance each time)
# - first shift based on sensor average location
# - two consecutive rotations for better alignment

X_11 = []
X_12 = []
Y_11 = []
Y_12 = []
X_21 = []
X_22 = []
Y_21 = []
Y_22 = []
X_31 = []
X_32 = []
Y_31 = []
Y_32 = []
X_41 = []
X_42 = []
Y_41 = []
Y_42 = []

print("Walking index: ", walking_indexes)
print("Distance total: ", distance_total)
for r in walking_indexes:
    #index_nr = r[1]
    sensor_number = r[0]
    for t in range(min_walking_distance):
        index_nr = r[1]+t
        if (index_nr + 1) < len(distance_total):
            step = distance_total[index_nr+1][sensor] - distance_total[index_nr][sensor]
            #step = 12
            for i in range(4):
                def f(x):
                    # APPROX: STEP LENGTH AS OBTAINED BEFORE
                    f0 = pow((x[0]), 2) + pow(x[1], 2) - pow(distance_total[index_nr][i], 2)
                    f1 = pow((step - x[0]), 2) + pow(x[1], 2) - pow(distance_total[index_nr + 1][i], 2)

                    return np.array([f0, f1])

                x0_1 = np.array([1000, 1000])
                x0_2 = np.array([-1000, -1000])
                solution_1 = fsolve(f, x0_1)
                solution_2 = fsolve(f, x0_2)
                print("\n", solution_1)
                print("\n", solution_2)

                if i == 0:

                    X_11.append(solution_1[0])
                    X_12.append(solution_2[0])
                    Y_11.append(solution_1[1])
                    Y_12.append(solution_2[1])

                if i == 1:

                    X_21.append(solution_1[0])
                    X_22.append(solution_2[0])
                    Y_21.append(solution_1[1])
                    Y_22.append(solution_2[1])

                if i == 2:

                    X_31.append(solution_1[0])
                    X_32.append(solution_2[0])
                    Y_31.append(solution_1[1])
                    Y_32.append(solution_2[1])

                if i == 3:

                    X_41.append(solution_1[0])
                    X_42.append(solution_2[0])
                    Y_41.append(solution_1[1])
                    Y_42.append(solution_2[1])

plt.plot(X_11, Y_11, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_21, Y_21, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_31, Y_31, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_41, Y_41, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(0, 0, marker='x', markerfacecolor='black', markersize=6)
plt.title("only proper indexes")

"""for t in range(len(X_11)):
    plt.plot(X_11[t], Y_11[t], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X_21[t], Y_21[t], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X_31[t], Y_31[t], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X_41[t], Y_41[t], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(0, 0, marker='x', markerfacecolor='black', markersize=6)
    plt.title("only proper indexes")
    plt.show()"""

# Now perform shift and rotation to align the sensor measurements...
# DEFINE THE ROTATION FUNCTION
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

# DEFINE THE ANGLE TO ROTATE BY

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


X11_shift = []
X12_shift = []
Y11_shift = []
Y12_shift = []

X21_shift = []
X22_shift = []
Y21_shift = []
Y22_shift = []

X31_shift = []
X32_shift = []
Y31_shift = []
Y32_shift = []

X41_shift = []
X42_shift = []
Y41_shift = []
Y42_shift = []

x_shift = 0
y_shift = 0

print(X_11)
print(X_11)
print(X_41)
print(X_41)

X_avg = (X_11[0] + X_21[0] + X_31[0] + X_41[0])/4
Y_avg = (Y_11[0] + Y_21[0] + Y_31[0] + Y_41[0])/4

t = 0
while t < len(X_11):
    x_shift = (X_11[t] + X_21[t] + X_31[t] + X_41[t])/4 - X_avg
    y_shift = (Y_11[t] + Y_21[t] + Y_31[t] + Y_41[t])/4 - Y_avg

    X11_shift.append(X_11[t] - x_shift)
    Y11_shift.append(Y_11[t] - y_shift)

    X21_shift.append(X_21[t] - x_shift)
    Y21_shift.append(Y_21[t] - y_shift)

    X31_shift.append(X_31[t] - x_shift)
    Y31_shift.append(Y_31[t] - y_shift)

    X41_shift.append(X_41[t] - x_shift)
    Y41_shift.append(Y_41[t] - y_shift)

    t = t + 1

plt.plot(X11_shift, Y11_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_shift, Y21_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_shift, Y31_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_shift, Y41_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.title("FOURTH method - shift only")
plt.savefig("4_shif.png")
plt.show()

# NOW ROTATION BIT....
X11_rotate = []
X12_rotate = []
Y11_rotate = []
Y12_rotate = []

X21_rotate = []
X22_rotate = []
Y21_rotate = []
Y22_rotate = []

X31_rotate = []
X32_rotate = []
Y31_rotate = []
Y32_rotate = []

X41_rotate = []
X42_rotate = []
Y41_rotate = []
Y42_rotate = []

X11_rotate_2 = []
X12_rotate_2 = []
Y11_rotate_2 = []
Y12_rotate_2 = []

X21_rotate_2 = []
X22_rotate_2 = []
Y21_rotate_2 = []
Y22_rotate_2 = []

X31_rotate_2 = []
X32_rotate_2 = []
Y31_rotate_2 = []
Y32_rotate_2 = []

X41_rotate_2 = []
X42_rotate_2 = []
Y41_rotate_2 = []
Y42_rotate_2 = []

for t in range(len(X11_shift)):
    origin = (X_avg, Y_avg)
    point = (X11_shift[0], Y11_shift[0])
    alpha = getAngle((X11_shift[t], Y11_shift[t]), origin, point)


    X11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[0])
    Y11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[1])

    X21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[0])
    Y21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[1])

    X31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[0])
    Y31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[1])

    X41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[0])
    Y41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[1])

    a = getAngle((X21_shift[t], Y21_shift[t]), origin, (X21_shift[0], Y21_shift[0]))
    b = getAngle((X31_shift[t], Y31_shift[t]), origin, (X31_shift[0], Y31_shift[0]))
    c = getAngle((X41_shift[t], Y41_shift[t]), origin, (X41_shift[0], Y41_shift[0]))

    beta  = 0.5 * max(a, b, c)

    X11_rotate_2.append(rotate(origin, (X11_shift[t], Y11_shift[t]), -math.radians(beta))[0])
    Y11_rotate_2.append(rotate(origin, (X11_shift[t], Y11_shift[t]), -math.radians(beta))[1])

    X21_rotate_2.append(rotate(origin, (X21_shift[t], Y21_shift[t]), -math.radians(beta))[0])
    Y21_rotate_2.append(rotate(origin, (X21_shift[t], Y21_shift[t]), -math.radians(beta))[1])

    X31_rotate_2.append(rotate(origin, (X31_shift[t], Y31_shift[t]), -math.radians(beta))[0])
    Y31_rotate_2.append(rotate(origin, (X31_shift[t], Y31_shift[t]), -math.radians(beta))[1])

    X41_rotate_2.append(rotate(origin, (X41_shift[t], Y41_shift[t]), -math.radians(alpha))[0])
    Y41_rotate_2.append(rotate(origin, (X41_shift[t], Y41_shift[t]), -math.radians(alpha))[1])

plt.plot(X11_rotate_2, Y11_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_rotate_2, Y21_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_rotate_2, Y31_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_rotate_2, Y41_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.title("FOURTH method - shift and double rotation")
plt.savefig("4_rotation12.png")
plt.show()


"""X1_avg_4 = sum(X11_rotate_2)/len(X11_rotate_2)
X2_avg_4 = sum(X21_rotate_2)/len(X21_rotate_2)
X3_avg_4 = sum(X31_rotate_2)/len(X31_rotate_2)
X4_avg_4 = sum(X41_rotate_2)/len(X41_rotate_2)
Y1_avg_4 = sum(Y11_rotate_2)/len(Y11_rotate_2)
Y2_avg_4 = sum(Y21_rotate_2)/len(Y21_rotate_2)
Y3_avg_4 = sum(Y31_rotate_2)/len(Y31_rotate_2)
Y4_avg_4 = sum(Y41_rotate_2)/len(Y41_rotate_2)"""

X1_avg_4 = sum(X11_rotate)/len(X11_rotate)
X2_avg_4 = sum(X21_rotate)/len(X21_rotate)
X3_avg_4 = sum(X31_rotate)/len(X31_rotate)
X4_avg_4 = sum(X41_rotate)/len(X41_rotate)
Y1_avg_4 = sum(Y11_rotate)/len(Y11_rotate)
Y2_avg_4 = sum(Y21_rotate)/len(Y21_rotate)
Y3_avg_4 = sum(Y31_rotate)/len(Y31_rotate)
Y4_avg_4 = sum(Y41_rotate)/len(Y41_rotate)

plt.plot(X1_avg_4, Y1_avg_4, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X2_avg_4, Y2_avg_4, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X3_avg_4, Y3_avg_4, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X4_avg_4, Y4_avg_4, marker='o', markerfacecolor='blue', markersize=6)

plt.annotate("S1", (X1_avg_4, Y1_avg_4))
plt.annotate("S2", (X2_avg_4, Y2_avg_4))
plt.annotate("S3", (X3_avg_4, Y3_avg_4))
plt.annotate("S4", (X4_avg_4, Y4_avg_4))

plt.title("FOURTH method - Average sensor locations")
plt.savefig("4_avg.png")
plt.show()


# FIFTH METHOD: SELECTION METHOD
# - use only walking indexes (straight line condition)
# - constant step = 12
# - first shift based on sensor average location
# - two consecutive rotations for better alignment

X_11 = []
X_12 = []
Y_11 = []
Y_12 = []
X_21 = []
X_22 = []
Y_21 = []
Y_22 = []
X_31 = []
X_32 = []
Y_31 = []
Y_32 = []
X_41 = []
X_42 = []
Y_41 = []
Y_42 = []

print("Walking index: ", walking_indexes)
print("Distance total: ", distance_total)
for r in walking_indexes:
    #index_nr = r[1]
    sensor_number = r[0]
    for t in range(min_walking_distance):
        index_nr = r[1]+t
        if (index_nr + 1) < len(distance_total):
            #step = distance_total[index_nr+1][sensor] - distance_total[index_nr][sensor]
            step = 12
            for i in range(4):
                def f(x):
                    # APPROX: STEP LENGTH AS OBTAINED BEFORE
                    f0 = pow((x[0]), 2) + pow(x[1], 2) - pow(distance_total[index_nr][i], 2)
                    f1 = pow((step - x[0]), 2) + pow(x[1], 2) - pow(distance_total[index_nr + 1][i], 2)

                    return np.array([f0, f1])

                x0_1 = np.array([1000, 1000])
                x0_2 = np.array([-1000, -1000])
                solution_1 = fsolve(f, x0_1)
                solution_2 = fsolve(f, x0_2)
                print("\n", solution_1)
                print("\n", solution_2)

                if i == 0:

                    X_11.append(solution_1[0])
                    X_12.append(solution_2[0])
                    Y_11.append(solution_1[1])
                    Y_12.append(solution_2[1])

                if i == 1:

                    X_21.append(solution_1[0])
                    X_22.append(solution_2[0])
                    Y_21.append(solution_1[1])
                    Y_22.append(solution_2[1])

                if i == 2:

                    X_31.append(solution_1[0])
                    X_32.append(solution_2[0])
                    Y_31.append(solution_1[1])
                    Y_32.append(solution_2[1])

                if i == 3:

                    X_41.append(solution_1[0])
                    X_42.append(solution_2[0])
                    Y_41.append(solution_1[1])
                    Y_42.append(solution_2[1])

plt.plot(X_11, Y_11, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_21, Y_21, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_31, Y_31, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_41, Y_41, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(0, 0, marker='x', markerfacecolor='black', markersize=6)
plt.title("only proper indexes")

"""for t in range(len(X_11)):
    plt.plot(X_11[t], Y_11[t], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X_21[t], Y_21[t], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X_31[t], Y_31[t], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X_41[t], Y_41[t], marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(0, 0, marker='x', markerfacecolor='black', markersize=6)
    plt.title("only proper indexes")
    plt.show()"""


X11_shift = []
X12_shift = []
Y11_shift = []
Y12_shift = []

X21_shift = []
X22_shift = []
Y21_shift = []
Y22_shift = []

X31_shift = []
X32_shift = []
Y31_shift = []
Y32_shift = []

X41_shift = []
X42_shift = []
Y41_shift = []
Y42_shift = []

x_shift = 0
y_shift = 0

print(X_11)
print(X_11)
print(X_41)
print(X_41)

X_avg = (X_11[0] + X_21[0] + X_31[0] + X_41[0])/4
Y_avg = (Y_11[0] + Y_21[0] + Y_31[0] + Y_41[0])/4

t = 0
while t < len(X_11):
    x_shift = (X_11[t] + X_21[t] + X_31[t] + X_41[t])/4 - X_avg
    y_shift = (Y_11[t] + Y_21[t] + Y_31[t] + Y_41[t])/4 - Y_avg

    X11_shift.append(X_11[t] - x_shift)
    Y11_shift.append(Y_11[t] - y_shift)

    X21_shift.append(X_21[t] - x_shift)
    Y21_shift.append(Y_21[t] - y_shift)

    X31_shift.append(X_31[t] - x_shift)
    Y31_shift.append(Y_31[t] - y_shift)

    X41_shift.append(X_41[t] - x_shift)
    Y41_shift.append(Y_41[t] - y_shift)

    t = t + 1

plt.plot(X11_shift, Y11_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_shift, Y21_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_shift, Y31_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_shift, Y41_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.title("FIFTH method - shift only")
plt.savefig("5_shif.png")
plt.show()

# NOW ROTATION BIT....
X11_rotate = []
X12_rotate = []
Y11_rotate = []
Y12_rotate = []

X21_rotate = []
X22_rotate = []
Y21_rotate = []
Y22_rotate = []

X31_rotate = []
X32_rotate = []
Y31_rotate = []
Y32_rotate = []

X41_rotate = []
X42_rotate = []
Y41_rotate = []
Y42_rotate = []

X11_rotate_2 = []
X12_rotate_2 = []
Y11_rotate_2 = []
Y12_rotate_2 = []

X21_rotate_2 = []
X22_rotate_2 = []
Y21_rotate_2 = []
Y22_rotate_2 = []

X31_rotate_2 = []
X32_rotate_2 = []
Y31_rotate_2 = []
Y32_rotate_2 = []

X41_rotate_2 = []
X42_rotate_2 = []
Y41_rotate_2 = []
Y42_rotate_2 = []

for t in range(len(X11_shift)):
    origin = (X_avg, Y_avg)
    point = (X11_shift[0], Y11_shift[0])
    alpha = getAngle((X11_shift[t], Y11_shift[t]), origin, point)


    X11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[0])
    Y11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[1])

    X21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[0])
    Y21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[1])

    X31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[0])
    Y31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[1])

    X41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[0])
    Y41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[1])

    a = getAngle((X21_shift[t], Y21_shift[t]), origin, (X21_shift[0], Y21_shift[0]))
    b = getAngle((X31_shift[t], Y31_shift[t]), origin, (X31_shift[0], Y31_shift[0]))
    c = getAngle((X41_shift[t], Y41_shift[t]), origin, (X41_shift[0], Y41_shift[0]))

    beta  = 0.5 * max(a, b, c)

    X11_rotate_2.append(rotate(origin, (X11_shift[t], Y11_shift[t]), -math.radians(beta))[0])
    Y11_rotate_2.append(rotate(origin, (X11_shift[t], Y11_shift[t]), -math.radians(beta))[1])

    X21_rotate_2.append(rotate(origin, (X21_shift[t], Y21_shift[t]), -math.radians(beta))[0])
    Y21_rotate_2.append(rotate(origin, (X21_shift[t], Y21_shift[t]), -math.radians(beta))[1])

    X31_rotate_2.append(rotate(origin, (X31_shift[t], Y31_shift[t]), -math.radians(beta))[0])
    Y31_rotate_2.append(rotate(origin, (X31_shift[t], Y31_shift[t]), -math.radians(beta))[1])

    X41_rotate_2.append(rotate(origin, (X41_shift[t], Y41_shift[t]), -math.radians(alpha))[0])
    Y41_rotate_2.append(rotate(origin, (X41_shift[t], Y41_shift[t]), -math.radians(alpha))[1])

plt.plot(X11_rotate_2, Y11_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_rotate_2, Y21_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_rotate_2, Y31_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_rotate_2, Y41_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.title("FIFTH method - shift and double rotation")
plt.savefig("5_rotation12.png")
plt.show()


"""X1_avg_5 = sum(X11_rotate_2)/len(X11_rotate_2)
X2_avg_5 = sum(X21_rotate_2)/len(X21_rotate_2)
X3_avg_5 = sum(X31_rotate_2)/len(X31_rotate_2)
X4_avg_5 = sum(X41_rotate_2)/len(X41_rotate_2)
Y1_avg_5 = sum(Y11_rotate_2)/len(Y11_rotate_2)
Y2_avg_5 = sum(Y21_rotate_2)/len(Y21_rotate_2)
Y3_avg_5 = sum(Y31_rotate_2)/len(Y31_rotate_2)
Y4_avg_5 = sum(Y41_rotate_2)/len(Y41_rotate_2)"""

X1_avg_5 = sum(X11_rotate)/len(X11_rotate)
X2_avg_5 = sum(X21_rotate)/len(X21_rotate)
X3_avg_5 = sum(X31_rotate)/len(X31_rotate)
X4_avg_5 = sum(X41_rotate)/len(X41_rotate)
Y1_avg_5 = sum(Y11_rotate)/len(Y11_rotate)
Y2_avg_5 = sum(Y21_rotate)/len(Y21_rotate)
Y3_avg_5 = sum(Y31_rotate)/len(Y31_rotate)
Y4_avg_5 = sum(Y41_rotate)/len(Y41_rotate)

plt.plot(X1_avg_5, Y1_avg_5, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X2_avg_5, Y2_avg_5, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X3_avg_5, Y3_avg_5, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X4_avg_5, Y4_avg_5, marker='o', markerfacecolor='blue', markersize=6)

plt.annotate("S1", (X1_avg_5, Y1_avg_5))
plt.annotate("S2", (X2_avg_5, Y2_avg_5))
plt.annotate("S3", (X3_avg_5, Y3_avg_5))
plt.annotate("S4", (X4_avg_5, Y4_avg_5))

plt.title("FIFTH method - average sensor locations")
plt.savefig("5_avg.png")
plt.show()

# --------------------

# PLOT ALL CALCULATED SENSOR POSITIONS WITH ALL METHODS
plt.plot(X1_avg_2, Y1_avg_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X2_avg_2, Y2_avg_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X3_avg_2, Y3_avg_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X4_avg_2, Y4_avg_2, marker='o', markerfacecolor='blue', markersize=6)

plt.plot(X1_avg_1, Y1_avg_1, marker='o', markerfacecolor='orange', markersize=6)
plt.plot(X2_avg_1, Y2_avg_1, marker='o', markerfacecolor='orange', markersize=6)
plt.plot(X3_avg_1, Y3_avg_1, marker='o', markerfacecolor='orange', markersize=6)
plt.plot(X4_avg_1, Y4_avg_1, marker='o', markerfacecolor='orange', markersize=6)

plt.plot(X1_avg_3, Y1_avg_3, marker='o', markerfacecolor='red', markersize=6)
plt.plot(X2_avg_3, Y2_avg_3, marker='o', markerfacecolor='red', markersize=6)
plt.plot(X3_avg_3, Y3_avg_3, marker='o', markerfacecolor='red', markersize=6)
plt.plot(X4_avg_3, Y4_avg_3, marker='o', markerfacecolor='red', markersize=6)

plt.plot(X1_avg_4, Y1_avg_4, marker='o', markerfacecolor='black', markersize=6)
plt.plot(X2_avg_4, Y2_avg_4, marker='o', markerfacecolor='black', markersize=6)
plt.plot(X3_avg_4, Y3_avg_4, marker='o', markerfacecolor='black', markersize=6)
plt.plot(X4_avg_4, Y4_avg_4, marker='o', markerfacecolor='black', markersize=6)

plt.plot(X1_avg_5, Y1_avg_5, marker='o', markerfacecolor='black', markersize=6)
plt.plot(X2_avg_5, Y2_avg_5, marker='o', markerfacecolor='black', markersize=6)
plt.plot(X3_avg_5, Y3_avg_5, marker='o', markerfacecolor='black', markersize=6)
plt.plot(X4_avg_5, Y4_avg_5, marker='o', markerfacecolor='black', markersize=6)


plt.annotate("S1_1", (X1_avg_1, Y1_avg_1))
plt.annotate("S2_1", (X2_avg_1, Y2_avg_1))
plt.annotate("S3_1", (X3_avg_1, Y3_avg_1))
plt.annotate("S4_1", (X4_avg_1, Y4_avg_1))

plt.annotate("S1_2", (X1_avg_2, Y1_avg_2))
plt.annotate("S2_2", (X2_avg_2, Y2_avg_2))
plt.annotate("S3_2", (X3_avg_2, Y3_avg_2))
plt.annotate("S4_2", (X4_avg_2, Y4_avg_2))

plt.annotate("S1_3", (X1_avg_3, Y1_avg_3))
plt.annotate("S2_3", (X2_avg_3, Y2_avg_3))
plt.annotate("S3_3", (X3_avg_3, Y3_avg_3))
plt.annotate("S4_3", (X4_avg_3, Y4_avg_3))

plt.annotate("S1_4", (X1_avg_4, Y1_avg_4))
plt.annotate("S2_4", (X2_avg_4, Y2_avg_4))
plt.annotate("S3_4", (X3_avg_4, Y3_avg_4))
plt.annotate("S4_4", (X4_avg_4, Y4_avg_4))

plt.annotate("S1_4", (X1_avg_5, Y1_avg_5))
plt.annotate("S2_4", (X2_avg_5, Y2_avg_5))
plt.annotate("S3_4", (X3_avg_5, Y3_avg_5))
plt.annotate("S4_4", (X4_avg_5, Y4_avg_5))

plt.title("All test methods - Average sensor locations")
plt.show()

# Now CORRECT THE DISTANCE DATA, adding a correction factor to the measurements
# ---------------------------------
# USE STEP = 12
step = 12

# FIRST CORRECTION MODEL
# adding 1/2 shoulder to any value (no further computation)
"""
distance = []
distance_total = []
shoulder = 50

for i in range(len(s1_avg)):
    #if s1_avg[1] != "nan":
    if not math.isnan(s1_avg[i]): #if not nan
        # Add 1/2 shoulder to correct measurement error
        distance.append(float(s1_avg[i])+shoulder/2)
        distance.append(float(s2_avg[i])+shoulder/2)
        distance.append(float(s3_avg[i])+shoulder/2)
        distance.append(float(s4_avg[i])+shoulder/2)
        distance_total.append(distance)
        distance = []

print(distance_total)
"""

# SECOND CORRECTION MODEL
#add 1/2 chest to any measurement
#add a further amount based on the difference between consecutive distances --> bigger variation corresponds to smaller addition and vice versa
"""
distance = []
distance_total = []
shoulder = 50

for i in range(len(s1_avg)-1):
    #if s1_avg[1] != "nan":
    if not math.isnan(s1_avg[i]): #if not nan

        # use:   correction= 40*(1-d/D)+10

        correction_1 = 40*(1-((s1_avg[i+1]-s1_avg[i])/step))+10
        correction_2 = 40 * (1 - ((s2_avg[i + 1] - s2_avg[i]) / step)) + 10
        correction_3 = 40 * (1 - ((s3_avg[i + 1] - s3_avg[i]) / step)) + 10
        correction_4 = 40 * (1 - ((s4_avg[i + 1] - s4_avg[i]) / step)) + 10

        distance.append(float(s1_avg[i])+correction_1)
        distance.append(float(s2_avg[i])+correction_2)
        distance.append(float(s3_avg[i])+correction_3)
        distance.append(float(s4_avg[i])+correction_4)
        distance_total.append(distance)
        distance = []

print(distance_total)
"""

# THIRD CORRECTION MODEL !!!
# fill in distance_total with normal values first
distance = []
distance_total = []
shoulder = 50

for i in range(len(s1_avg)-1):
    if not math.isnan(s1_avg[i]): #if not nan
        distance.append(float(s1_avg[i]))
        distance.append(float(s2_avg[i]))
        distance.append(float(s3_avg[i]))
        distance.append(float(s4_avg[i]))
        distance_total.append(distance)
        distance = []

# function to determine the angle of the direction
def angle(x1, y1, x2, y2):
    d = math.sqrt((x1-x2)**2+(y1-y2)**2)
    c = abs(y1)
    if d <= 2*step:
        #alpha = math.asin(0.5*d/step)
        alpha = math.asin(c/step)
        # NB value id rads
        return alpha
    else:
        return math.pi/2

for r in range(len(distance_total)-1):
    #if s1_avg[1] != "nan":
    if not math.isnan(s1_avg[i]): #if not nan
        if r not in still_indexes:
            for i in range(4):
                def f(x):
                    # APPROX: STEP LENGTH AS OBTAINED BEFORE
                    f0 = pow((x[0]), 2) + pow(x[1], 2) - pow(distance_total[r+1][i], 2)
                    f1 = pow((distance_total[r][i] - x[0]), 2) + pow(x[1], 2) - pow(step, 2)

                    return np.array([f0, f1])

                x0_1 = np.array([distance_total[r+1][i], step])
                x0_2 = np.array([distance_total[r+1][i], -step])
                solution_1 = fsolve(f, x0_1)
                solution_2 = fsolve(f, x0_2)
                """
                print("\n", solution_1)
                print("\n", solution_2)
                """
                alpha = angle(float(solution_1[0]), float(solution_2[0]), float(solution_1[1]), float(solution_2[1]))
                # NB value in rads
                if alpha < 0.433:
                    correction = 10/math.cos(alpha)
                elif alpha > 0.433:
                    beta = math.pi/2 - alpha
                    correction = 50/math.cos(beta)

                    distance_total[r][i] = distance_total[r][i] + correction


print ("Test new correction method !!!!!")
print(distance_total)

plot_1 = []
plot_2 = []
plot_3 = []
plot_4 = []

for i in range(len(distance_total)):
    plot_1.append(distance_total[i][0])
    plot_2.append(distance_total[i][1])
    plot_3.append(distance_total[i][2])
    plot_4.append(distance_total[i][3])

plt.plot(plot_1)
plt.plot(plot_2)
plt.plot(plot_3)
plt.plot(plot_4)
plt.title("Data with new correction method")
plt.show()


# TEST NEW CORRECTION MODEL
# ----------------------------
# Define new walking_indexes array
walking_indexes = []
for i in range(len(distance_total)):
    if i not in still_indexes:
        walking_indexes.append(i)


X_11 = []
X_12 = []
Y_11 = []
Y_12 = []
X_21 = []
X_22 = []
Y_21 = []
Y_22 = []
X_31 = []
X_32 = []
Y_31 = []
Y_32 = []
X_41 = []
X_42 = []
Y_41 = []
Y_42 = []



for r in walking_indexes:
    if (r + 1) < len(distance_total):
        step = 12
        for i in range(4):
            def f(x):
                # APPROX: STEP LENGTH AS OBTAINED BEFORE
                f0 = pow((x[0]), 2) + pow(x[1], 2) - pow(distance_total[r][i], 2)
                f1 = pow((step - x[0]), 2) + pow(x[1], 2) - pow(distance_total[r + 1][i], 2)

                return np.array([f0, f1])

            x0_1 = np.array([1000, 1000])
            x0_2 = np.array([-1000, -1000])
            solution_1 = fsolve(f, x0_1)
            solution_2 = fsolve(f, x0_2)
            print("\n", solution_1)
            print("\n", solution_2)

            if i == 0:

                X_11.append(solution_1[0])
                X_12.append(solution_2[0])
                Y_11.append(solution_1[1])
                Y_12.append(solution_2[1])

            if i == 1:

                X_21.append(solution_1[0])
                X_22.append(solution_2[0])
                Y_21.append(solution_1[1])
                Y_22.append(solution_2[1])

            if i == 2:

                X_31.append(solution_1[0])
                X_32.append(solution_2[0])
                Y_31.append(solution_1[1])
                Y_32.append(solution_2[1])

            if i == 3:

                X_41.append(solution_1[0])
                X_42.append(solution_2[0])
                Y_41.append(solution_1[1])
                Y_42.append(solution_2[1])

plt.plot(X_11, Y_11, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_21, Y_21, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_31, Y_31, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X_41, Y_41, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(0, 0, marker='x', markerfacecolor='black', markersize=6)
plt.title("SOLUTIONS WITH NEW CORRECTION MODEL")
plt.show()

# SHIFT PART, NEW CORRECTION MODEL

X11_shift = []
X12_shift = []
Y11_shift = []
Y12_shift = []

X21_shift = []
X22_shift = []
Y21_shift = []
Y22_shift = []

X31_shift = []
X32_shift = []
Y31_shift = []
Y32_shift = []

X41_shift = []
X42_shift = []
Y41_shift = []
Y42_shift = []

x_shift = 0
y_shift = 0

X_avg = (X_11[0] + X_21[0] + X_31[0] + X_41[0])/4
Y_avg = (Y_11[0] + Y_21[0] + Y_31[0] + Y_41[0])/4

t = 0
while t < len(X_11):
    x_shift = (X_11[t] + X_21[t] + X_31[t] + X_41[t])/4 - X_avg
    y_shift = (Y_11[t] + Y_21[t] + Y_31[t] + Y_41[t])/4 - Y_avg

    X11_shift.append(X_11[t] - x_shift)
    Y11_shift.append(Y_11[t] - y_shift)

    X21_shift.append(X_21[t] - x_shift)
    Y21_shift.append(Y_21[t] - y_shift)

    X31_shift.append(X_31[t] - x_shift)
    Y31_shift.append(Y_31[t] - y_shift)

    X41_shift.append(X_41[t] - x_shift)
    Y41_shift.append(Y_41[t] - y_shift)

    t = t + 1

plt.plot(X11_shift, Y11_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_shift, Y21_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_shift, Y31_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_shift, Y41_shift, marker='o', markerfacecolor='blue', markersize=6)
plt.title("SHIFT, NEW CORRECTION MODEL")
plt.show()

# ROTATION PART, NEW CORRECTION MODEL
X11_rotate = []
X12_rotate = []
Y11_rotate = []
Y12_rotate = []

X21_rotate = []
X22_rotate = []
Y21_rotate = []
Y22_rotate = []

X31_rotate = []
X32_rotate = []
Y31_rotate = []
Y32_rotate = []

X41_rotate = []
X42_rotate = []
Y41_rotate = []
Y42_rotate = []

X11_rotate_2 = []
X12_rotate_2 = []
Y11_rotate_2 = []
Y12_rotate_2 = []

X21_rotate_2 = []
X22_rotate_2 = []
Y21_rotate_2 = []
Y22_rotate_2 = []

X31_rotate_2 = []
X32_rotate_2 = []
Y31_rotate_2 = []
Y32_rotate_2 = []

X41_rotate_2 = []
X42_rotate_2 = []
Y41_rotate_2 = []
Y42_rotate_2 = []

for t in range(len(X11_shift)):
    origin = (X_avg, Y_avg)
    point = (X11_shift[0], Y11_shift[0])
    alpha = getAngle((X11_shift[t], Y11_shift[t]), origin, point)



    X11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[0])
    Y11_rotate.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(alpha))[1])

    X21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[0])
    Y21_rotate.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(alpha))[1])

    X31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[0])
    Y31_rotate.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(alpha))[1])

    X41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[0])
    Y41_rotate.append(rotate(origin, (X41_shift[t], Y41_shift[t]), math.radians(alpha))[1])

    a = getAngle((X21_shift[t], Y21_shift[t]), origin, (X21_shift[0], Y21_shift[0]))
    b = getAngle((X31_shift[t], Y31_shift[t]), origin, (X31_shift[0], Y31_shift[0]))
    c = getAngle((X41_shift[t], Y41_shift[t]), origin, (X41_shift[0], Y41_shift[0]))

    beta  = 0.5 * max(a, b, c)

    beta = getAngle((X31_shift[t], Y31_shift[t]), origin, (X31_shift[0], Y31_shift[0]))
    if beta > 90:
        beta = (360 - 0.5*beta)
    else:
        beta = 0.5*beta


    X11_rotate_2.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(beta))[0])
    Y11_rotate_2.append(rotate(origin, (X11_shift[t], Y11_shift[t]), math.radians(beta))[1])

    X21_rotate_2.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(beta))[0])
    Y21_rotate_2.append(rotate(origin, (X21_shift[t], Y21_shift[t]), math.radians(beta))[1])

    X31_rotate_2.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(beta))[0])
    Y31_rotate_2.append(rotate(origin, (X31_shift[t], Y31_shift[t]), math.radians(beta))[1])

    X41_rotate_2.append(rotate(origin, (X41_shift[t], Y41_shift[t]), -math.radians(alpha))[0])
    Y41_rotate_2.append(rotate(origin, (X41_shift[t], Y41_shift[t]), -math.radians(alpha))[1])

plt.plot(X11_rotate_2, Y11_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X21_rotate_2, Y21_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X31_rotate_2, Y31_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X41_rotate_2, Y41_rotate_2, marker='o', markerfacecolor='blue', markersize=6)
plt.title("DOUBLE ROTATION, NEW CORRECTION MODEL")
plt.show()


"""X1_avg_6 = sum(X11_rotate_2)/len(X11_rotate_2)
X2_avg_6 = sum(X21_rotate_2)/len(X21_rotate_2)
X3_avg_6 = sum(X31_rotate_2)/len(X31_rotate_2)
X4_avg_6 = sum(X41_rotate_2)/len(X41_rotate_2)
Y1_avg_6 = sum(Y11_rotate_2)/len(Y11_rotate_2)
Y2_avg_6 = sum(Y21_rotate_2)/len(Y21_rotate_2)
Y3_avg_6 = sum(Y31_rotate_2)/len(Y31_rotate_2)
Y4_avg_6 = sum(Y41_rotate_2)/len(Y41_rotate_2)"""

X1_avg_6 = sum(X11_rotate)/len(X11_rotate)
X2_avg_6 = sum(X21_rotate)/len(X21_rotate)
X3_avg_6 = sum(X31_rotate)/len(X31_rotate)
X4_avg_6 = sum(X41_rotate)/len(X41_rotate)
Y1_avg_6 = sum(Y11_rotate)/len(Y11_rotate)
Y2_avg_6 = sum(Y21_rotate)/len(Y21_rotate)
Y3_avg_6 = sum(Y31_rotate)/len(Y31_rotate)
Y4_avg_6 = sum(Y41_rotate)/len(Y41_rotate)

plt.plot(X1_avg_6, Y1_avg_6, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X2_avg_6, Y2_avg_6, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X3_avg_6, Y3_avg_6, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(X4_avg_6, Y4_avg_6, marker='o', markerfacecolor='blue', markersize=6)

plt.annotate("S1", (X1_avg_6, Y1_avg_6))
plt.annotate("S2", (X2_avg_6, Y2_avg_6))
plt.annotate("S3", (X3_avg_6, Y3_avg_6))
plt.annotate("S4", (X4_avg_6, Y4_avg_6))

plt.title("AVG. SENSOR LOCATION, NEW CORRECTION MODEL")
plt.savefig("FINAL_AVG_SENSOR_LOCATION.PNG")
plt.show()

# --------------------

# TRY TO USE THE GIVEN POSITIONS OF THE SENSORS AND PLOT THE PATH USING +1/2 SHOULDER:
s1x=0
s1y=0

s4x=507.7
s4y=-310.8

s3x=749.2
s3y=313.8

s2x=507.7
s2y=630.7

plt.plot(s1x, s1y, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(s2x, s2y, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(s3x, s3y, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(s4x, s4y, marker='o', markerfacecolor='blue', markersize=6)

plt.annotate("S1", (s1x, s1y))
plt.annotate("S2", (s2x, s2y))
plt.annotate("S3", (s3x, s3y))
plt.annotate("S4", (s4x, s4y))

plt.title("Real sensor locations")
plt.savefig("Real_sensor_locations.png")
plt.show()

#s1-s2
path1x = []
path1y = []
path2x = []
path2y = []

#s3-s4
path3x = []
path3y = []
path4x = []
path4y = []

#s1-s4
path5x = []
path5y = []
path6x = []
path6y = []

#s2-s3
path7x = []
path7y = []
path8x = []
path8y = []

#s1-s3
path9x = []
path9y = []
path10x = []
path10y = []

for i in range(len(distance_total)):
    def f(x):
        f0 = pow(x[0] - s1x, 2) + pow(x[1] - s1y, 2) - pow(distance_total[i][0], 2)
        f1 = pow((x[0] - s2x), 2) + pow((x[1] - s2y), 2) - pow(distance_total[i][1], 2)

        return np.array([f0, f1])


    X_0 = X1_avg_1
    Y_0 = Y1_avg_1
    """X_0 = s1x
    Y_0 = s1y"""
    x0_1 = np.array([X_0, Y_0])
    x0_2 = np.array([-X_0, -Y_0])
    solution_1 = fsolve(f, x0_1)
    solution_2 = fsolve(f, x0_2)

    path1x.append(solution_1[0])
    path1y.append(solution_1[1])
    path2x.append(solution_2[0])
    path2y.append(solution_2[1])

for i in range(len(distance_total)):
    def f(x):
        f0 = pow(x[0] - s3x, 2) + pow(x[1] - s3y, 2) - pow(distance_total[i][2], 2)
        f1 = pow((x[0] - s4x), 2) + pow((x[1] - s4y), 2) - pow(distance_total[i][3], 2)

        return np.array([f0, f1])

    # With 1000 the proper solutions are found.... must use the outermost sensor values... still to do
    X_0 = 1000
    Y_0 = 1000
    """X_0 = X2_avg_1
    Y_0 = Y2_avg_1"""
    x0_1 = np.array([X_0, Y_0])
    x0_2 = np.array([-X_0, -Y_0])
    solution_1 = fsolve(f, x0_1)
    solution_2 = fsolve(f, x0_2)

    path3x.append(solution_1[0])
    path3y.append(solution_1[1])
    path4x.append(solution_2[0])
    path4y.append(solution_2[1])

for i in range(len(distance_total)):
    def f(x):
        f0 = pow(x[0] - s1x, 2) + pow(x[1] - s1y, 2) - pow(distance_total[i][0], 2)
        f1 = pow((x[0] - s4x), 2) + pow((x[1] - s4y), 2) - pow(distance_total[i][3], 2)

        return np.array([f0, f1])

    # With 1000 the proper solutions are found.... must use the outermost sensor values... still to do
    X_0 = 1000
    Y_0 = 1000
    """X_0 = X2_avg_1
    Y_0 = Y2_avg_1"""
    x0_1 = np.array([X_0, Y_0])
    x0_2 = np.array([-X_0, -Y_0])
    solution_1 = fsolve(f, x0_1)
    solution_2 = fsolve(f, x0_2)

    path5x.append(solution_1[0])
    path5y.append(solution_1[1])
    path6x.append(solution_2[0])
    path6y.append(solution_2[1])

for i in range(len(distance_total)):
    def f(x):
        f0 = pow(x[0] - s2x, 2) + pow(x[1] - s2y, 2) - pow(distance_total[i][1], 2)
        f1 = pow((x[0] - s3x), 2) + pow((x[1] - s3y), 2) - pow(distance_total[i][2], 2)

        return np.array([f0, f1])

    # With 1000 the proper solutions are found.... must use the outermost sensor values... still to do
    X_0 = 1000
    Y_0 = 1000
    """X_0 = X2_avg_1
    Y_0 = Y2_avg_1"""
    x0_1 = np.array([X_0, Y_0])
    x0_2 = np.array([-X_0, -Y_0])
    solution_1 = fsolve(f, x0_1)
    solution_2 = fsolve(f, x0_2)

    path7x.append(solution_1[0])
    path7y.append(solution_1[1])
    path8x.append(solution_2[0])
    path8y.append(solution_2[1])

for i in range(len(distance_total)):
    def f(x):
        f0 = pow(x[0] - s1x, 2) + pow(x[1] - s1y, 2) - pow(distance_total[i][0], 2)
        f1 = pow((x[0] - s3x), 2) + pow((x[1] - s3y), 2) - pow(distance_total[i][2], 2)

        return np.array([f0, f1])

    # With 1000 the proper solutions are found.... must use the outermost sensor values... still to do
    X_0 = 1000
    Y_0 = 1000
    """X_0 = X2_avg_1
    Y_0 = Y2_avg_1"""
    x0_1 = np.array([X_0, Y_0])
    x0_2 = np.array([-X_0, -Y_0])
    solution_1 = fsolve(f, x0_1)
    solution_2 = fsolve(f, x0_2)

    path9x.append(solution_1[0])
    path9y.append(solution_1[1])
    path10x.append(solution_2[0])
    path10y.append(solution_2[1])

#s1-s2
plt.plot(path1x, path1y, marker='o', markerfacecolor='red', markersize=3)
plt.plot(path2x, path2y, marker='o', markerfacecolor='red', markersize=3)
#s3-s4
plt.plot(path3x, path3y, marker='o', markerfacecolor='red', markersize=3)
plt.plot(path4x, path4y, marker='o', markerfacecolor='red', markersize=3)
#s1-s4
plt.plot(path5x, path5y, marker='o', markerfacecolor='red', markersize=3)
plt.plot(path6x, path6y, marker='o', markerfacecolor='red', markersize=3)
#s2-s3
plt.plot(path7x, path7y, marker='o', markerfacecolor='red', markersize=3)
plt.plot(path8x, path8y, marker='o', markerfacecolor='red', markersize=3)
#s1-s3
plt.plot(path9x, path9y, marker='o', markerfacecolor='red', markersize=3)
plt.plot(path10x, path10y, marker='o', markerfacecolor='red', markersize=3)

plt.plot(s1x, s1y, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(s2x, s2y, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(s3x, s3y, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(s4x, s4y, marker='o', markerfacecolor='blue', markersize=6)

plt.annotate("S1", (s1x, s1y))
plt.annotate("S2", (s2x, s2y))
plt.annotate("S3", (s3x, s3y))
plt.annotate("S4", (s4x, s4y))

plt.title("All paths for couples of sensors - with corrected distances")
plt.savefig("Paths_corrected_distances")
plt.show()

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# OTHER WORK TO TEST ALGORITHM...
# PLOTTING INTERSECTIONS AT EACH TIME


#print("LINES: ", lines)
# -.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..-
# -.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..-
# -.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..-

# PLOT CIRCLES WITH ERROR RANGE BOTH UP AND DOWN - USE SENSOR POSITION 1

for t in range(len(distance_total)):
    circle1 = plt.Circle((X1_avg_1, Y1_avg_1), distance_total[t][0], color='b', fill=False, linestyle='-')
    circle2 = plt.Circle((X2_avg_1, Y2_avg_1), distance_total[t][1], color='r', fill=False, linestyle='-')
    circle3 = plt.Circle((X3_avg_1, Y3_avg_1), distance_total[t][2], color='g', fill=False, linestyle='-')
    circle4 = plt.Circle((X4_avg_1, Y4_avg_1), distance_total[t][3], color='y', fill=False, linestyle='-')

    circle5 = plt.Circle((X1_avg_1, Y1_avg_1), distance_total[t][0]+shoulder, color='b', fill=False, linestyle='--')
    circle6 = plt.Circle((X2_avg_1, Y2_avg_1), distance_total[t][1]+shoulder, color='r', fill=False, linestyle='--')
    circle7 = plt.Circle((X3_avg_1, Y3_avg_1), distance_total[t][2]+shoulder, color='g', fill=False, linestyle='--')
    circle8 = plt.Circle((X4_avg_1, Y4_avg_1), distance_total[t][3]+shoulder, color='y', fill=False, linestyle='--')

    circle9 = plt.Circle((X1_avg_1, Y1_avg_1), distance_total[t][0] - shoulder, color='b', fill=False, linestyle='--')
    circle10 = plt.Circle((X2_avg_1, Y2_avg_1), distance_total[t][1] - shoulder, color='r', fill=False, linestyle='--')
    circle11 = plt.Circle((X3_avg_1, Y3_avg_1), distance_total[t][2] - shoulder, color='g', fill=False, linestyle='--')
    circle12 = plt.Circle((X4_avg_1, Y4_avg_1), distance_total[t][3] - shoulder, color='y', fill=False, linestyle='--')

    ax = plt.gca()
    ax.cla()  # clear things for fresh plot
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)
    ax.add_patch(circle5)
    ax.add_patch(circle6)
    ax.add_patch(circle7)
    ax.add_patch(circle8)
    ax.add_patch(circle9)
    ax.add_patch(circle10)
    ax.add_patch(circle11)
    ax.add_patch(circle12)

    plt.plot(X1_avg_1, Y1_avg_1, marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X2_avg_1, Y2_avg_1, marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X3_avg_1, Y3_avg_1, marker='o', markerfacecolor='blue', markersize=6)
    plt.plot(X4_avg_1, Y4_avg_1, marker='o', markerfacecolor='blue', markersize=6)

    plt.annotate("S1", (X1_avg_1, Y1_avg_1))
    plt.annotate("S2", (X2_avg_1, Y2_avg_1))
    plt.annotate("S3", (X3_avg_1, Y3_avg_1))
    plt.annotate("S4", (X4_avg_1, Y4_avg_1))

    plt.title("Solution 1 and cirles")
    plt.show()

# -.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..-
# -.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..-
# -.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..--.-.-.-.-.-..-
