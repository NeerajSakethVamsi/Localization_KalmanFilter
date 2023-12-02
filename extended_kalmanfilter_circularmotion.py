import numpy as np
import matplotlib.pyplot as plt
import math

class Filter:
    def __init__(self, var=None):
        self.filter = var
        self.filter.predict()

    def get_point(self):
        return self.filter.x[0:2].flatten()

    def predict_point(self, measurement):

        self.filter.predict()
        self.filter.update(measurement)
        return self.get_point()


class ExtendedKalmanFilter:
    def __init__(self,sigma,u,dt):
        self.x = np.array([0.,  # x
                           0.,  # y
                           0.,  # orientation
                           0.]).reshape(-1, 1)  # linear velocity

        self.u = u
        self.dt = dt
        self.p = np.diag([999. for i in range(4)])
        self.q = np.diag([0.1 for i in range(4)])
        self.r = np.diag([sigma, sigma])
        self.jf = np.diag([1. for i in range(4)])
        self.jg = np.array([[1., 0., 0., 0.],
                            [0., 1., 0., 0.]])
        self.h = np.array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.]])
        self.z = np.array([0.,
                           0.]).reshape(-1, 1)

    def predict(self):
        x, y, phi, v = self.x
        vt, wt = self.u
        dT = self.dt

        new_x = x + vt * np.cos(phi) * dT
        new_y = y + vt * np.sin(phi) * dT
        new_phi = phi + wt * dT
        new_v = vt

        self.x[0]=new_x
        self.x[1]=new_y
        self.x[2]=new_phi
        self.x[3]=new_v

        self.jf[0][2] = -v * np.sin(phi) * dT
        self.jf[0][3] = v * np.cos(phi) * dT
        self.jf[1][2] = v * np.cos(phi) * dT
        self.jf[1][3] = v * np.sin(phi) * dT

        # p-prediction:
        self.p = self.jf.dot(self.p).dot(self.jf.transpose()) + self.q

    def update(self, measurement):
        z = np.array(measurement).reshape(-1, 1)
        y = z - self.h.dot(self.x)

        s = self.jg.dot(self.p).dot(self.jg.transpose()) + self.r
        k = self.p.dot(self.jg.transpose()).dot(np.linalg.inv(s))

        self.x = self.x + k.dot(y)
        self.p = (np.eye(4) - k.dot(self.h)).dot(self.p)

def action(parfilter, measurements):
    result = []
    for i in range(len(measurements)):
        m = measurements[i]
        pred = parfilter.predict_point(m)
        result.append(pred)

    return result

def create_artificial_circle_data(radius,dt,u,sigma):
    measurements = []
    vt, wt = u
    phi = np.arange(0, 2*math.pi, wt * dt)
    x = radius * np.cos(phi) + sigma * np.random.randn(len(phi))
    y = radius * np.sin(phi) + sigma * np.random.randn(len(phi))

    measurements += [[x, y] for x, y in zip(x, y)]
    return measurements

def plot(measurements, result, deadreckon_results):
    plt.figure("Extended Kalman Filter Visualization")
    plt.scatter([x[0] for x in measurements], [x[1] for x in measurements], c='red', label='GPS Position', alpha=0.3,s=4)
    plt.scatter([x[0] for x in result], [x[1] for x in result], c='blue', label='Kalman filter estimate Position', alpha=0.3, s=4,marker='x')
    plt.scatter([x[0] for x in deadreckon_results], [x[1] for x in deadreckon_results], c='black', label='dead reckoning Position', alpha=0.3, s=4,marker='x')
    plt.legend()
    plt.grid('on')
    plt.title("Circular Motion Robot and Kalman Filter Prediction")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.tight_layout()
    plt.show()

def deadReckoning(robo, u, dt):
    initial_state = robo.x
    vt, wt = u
    x=vt/wt
    y=0
    phi=math.pi/2
    x_list=[]
    y_list=[]
    phi_list=[]
    result = []
    T = 2 * math.pi / wt
    for i in range(int(T/dt)):
        new_x = x + vt * np.cos(phi) * dt
        new_y = y + vt * np.sin(phi) * dt
        new_phi = phi + wt * dt
        x = new_x
        y = new_y
        phi = new_phi
        # new_v = vt

        x_list.append(new_x)
        y_list.append(new_y)
        phi_list.append(new_phi)
    result += [[p, q] for p, q in zip(x_list, y_list)]

    return result

if __name__ == '__main__':
    input_u = [10, 2]
    radiusOfCircle = input_u[0] / input_u[1]
    robot = ExtendedKalmanFilter(sigma=0.1, u=input_u, dt=0.001)
    robotController = Filter(robot)
    deadreckon_results=deadReckoning(robo=robot, u=input_u, dt=0.001)
    measurementGPS = create_artificial_circle_data(radius=radiusOfCircle,u=input_u, dt=0.001, sigma=0.1)
    results = action(robotController, measurementGPS)
    plot(measurementGPS, results, deadreckon_results)
