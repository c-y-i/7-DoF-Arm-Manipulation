import numpy as np
from math import pi

class FK():
    def __init__(self):
        self.d = [0.192, 0, 0.121 + 0.195, 0, 0.125 + 0.259, 0, 0.159 + 0.051]
        self.a = [0, 0, 0.0825, 0.0825, 0, 0.088, 0]
        self.alpha = [-pi/2, pi/2, pi/2, pi/2, -pi/2, pi/2, 0]

    def DH_transform(self, a, alpha, d, theta):
        DH_matrix = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        return DH_matrix

    def forward(self, q):
        q = q.astype(np.float64)
        q += np.array([0, 0, 0, pi, 0, -pi, -pi/4])
        
        T0e = np.identity(4)
        joint_offset_z = [0.141, 0.0, 0.195, 0.0, 0.125, -0.015, 0.051, 0.0]
        T0e[2, 3] = joint_offset_z[0]
        
        jointPositions = np.zeros((8, 3))
        jointPositions[0] = T0e[:3, 3]
        
        for i in range(7):
            T_i = self.DH_transform(self.a[i], self.alpha[i], self.d[i], q[i])
            T0e = np.dot(T0e, T_i)
            jointPositions[i + 1] = T0e[:3, 3]

        return jointPositions, T0e

    def get_axis_of_rotation(self, q):
        z = []
        T0e = np.identity(4)
        q = q.astype(np.float64)
        q += np.array([0, 0, 0, pi, 0, -pi, -pi/4])
        
        for i in range(7):
            z.append(T0e[:3, 2])
            T_i = self.DH_transform(self.a[i], self.alpha[i], self.d[i], q[i])
            T0e = np.dot(T0e, T_i)
        
        return z
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        q = q.astype(np.float64)
        q += np.array([0, 0, 0, pi, 0, -pi, -pi/4])
        
        T0e = np.identity(4)
        Ai = []
        
        for i in range(7):
            T_i = self.DH_transform(self.a[i], self.alpha[i], self.d[i], q[i])
            T0e = np.dot(T0e, T_i)
            Ai.append(T0e.copy())
        
        return Ai
    
if __name__ == "__main__":

    fk = FK()
    np.set_printoptions(suppress=True)
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)

    z_product = fk.get_axis_of_rotation(q)

    print(f"Z: {z_product}")

    Ai = fk.compute_Ai(q)
    for i, A in enumerate(Ai):
        print(f"A[{i}]:\n{A}")