import numpy as np
import matplotlib.pyplot as plt

class triangle_dataset_generator:
    def __init__(self, n_sample=300):
        self._n_sample = n_sample
        
        # 정삼각형의 꼭짓점 좌표 계산
        self._vertices = np.array([
            [0, 1],           # 상단 꼭짓점
            [-np.sqrt(3)/2, -0.5],  # 좌하단 꼭짓점
            [np.sqrt(3)/2, -0.5]    # 우하단 꼭짓점
        ])

    def _is_inside_triangle(self, points):
        """주어진 점이 정삼각형 내부에 있는지 확인하는 함수"""
        
        def area(x1, y1, x2, y2, x3, y3):
            """세 점으로 이루어진 삼각형의 면적"""
            return abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2))/2.0)
        
        # 전체 정삼각형의 면적
        A = area(self._vertices[0,0], self._vertices[0,1],
                self._vertices[1,0], self._vertices[1,1],
                self._vertices[2,0], self._vertices[2,1])
        
        # 각 점에 대해 세 부분 삼각형의 면적 합 계산
        inside = np.zeros(len(points), dtype=bool)
        for i in range(len(points)):
            x, y = points[i]
            
            # 점 P와 삼각형의 각 변으로 만들어지는 세 삼각형의 면적
            A1 = area(x, y, self._vertices[1,0], self._vertices[1,1],
                     self._vertices[2,0], self._vertices[2,1])
            A2 = area(self._vertices[0,0], self._vertices[0,1],
                     x, y, self._vertices[2,0], self._vertices[2,1])
            A3 = area(self._vertices[0,0], self._vertices[0,1],
                     self._vertices[1,0], self._vertices[1,1], x, y)
            
            # 세 부분 삼각형의 면적 합이 전체 삼각형의 면적과 같으면 내부에 있는 점
            inside[i] = abs(A - (A1 + A2 + A3)) < 1e-10
            
        return inside

    def make_dataset(self):
        # 데이터 포인트 생성을 위한 범위 설정
        x_min, x_max = -2, 2
        y_min, y_max = -2, 2
        
        # 균일 분포에서 점들 생성
        x_data = np.random.uniform(x_min, x_max, (self._n_sample, 1))
        y_data = np.random.uniform(y_min, y_max, (self._n_sample, 1))
        points = np.hstack((x_data, y_data))
        
        # 각 점이 정삼각형 내부에 있는지 확인
        labels = self._is_inside_triangle(points).astype(int).reshape(-1, 1)
        
        # 최종 데이터셋 생성 (x0=1인 열 추가)
        x0 = np.ones((self._n_sample, 1))
        data = np.hstack((x0, points, labels))
        
        return data

    def plot_dataset(self, data=None):
        """생성된 데이터셋을 시각화하는 함수"""
        
        if data is None:
            data = self.make_dataset()
            
        # 데이터 포인트 분리
        positive = data[data[:, -1] == 1]
        negative = data[data[:, -1] == 0]
        
        plt.figure(figsize=(10, 10))
        
        # 정삼각형 그리기
        triangle = np.vstack((self._vertices, self._vertices[0]))  # 닫힌 다각형을 위해 첫 점 반복
        plt.plot(triangle[:, 0], triangle[:, 1], 'g-', label='Triangle Boundary')
        
        # 데이터 포인트 그리기
        plt.scatter(positive[:, 1], positive[:, 2], c='blue', marker='o', label='Inside (1)')
        plt.scatter(negative[:, 1], negative[:, 2], c='red', marker='x', label='Outside (0)')
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('Triangle Dataset')
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        
        plt.show()