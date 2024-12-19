import numpy as np
import matplotlib.pyplot as plt


class triangle_dataset_generator:
    def __init__(self, n_sample=300, size=1.0, noise=0.0):
        self._n_sample = n_sample
        self._size = size
        self._noise = noise

        # 정삼각형의 꼭짓점 좌표 계산
        self._vertices = np.array([
            [0, self._size],  # 상단 꼭짓점
            [-self._size * np.sqrt(3) / 2, -self._size * 0.5],  # 좌하단 꼭짓점
            [self._size * np.sqrt(3) / 2, -self._size * 0.5]  # 우하단 꼭짓점
        ])

    def _is_inside_triangle(self, points):
        """주어진 점이 정삼각형 내부에 있는지 확인하는 함수"""

        def area(x1, y1, x2, y2, x3, y3):
            """세 점으로 이루어진 삼각형의 면적"""
            return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

        # 전체 정삼각형의 면적
        A = area(self._vertices[0, 0], self._vertices[0, 1],
                 self._vertices[1, 0], self._vertices[1, 1],
                 self._vertices[2, 0], self._vertices[2, 1])

        # 각 점에 대해 세 부분 삼각형의 면적 합 계산
        inside = np.zeros(len(points), dtype=bool)
        for i in range(len(points)):
            x, y = points[i]

            # 점 P와 삼각형의 각 변으로 만들어지는 세 삼각형의 면적
            A1 = area(x, y, self._vertices[1, 0], self._vertices[1, 1],
                      self._vertices[2, 0], self._vertices[2, 1])
            A2 = area(self._vertices[0, 0], self._vertices[0, 1],
                      x, y, self._vertices[2, 0], self._vertices[2, 1])
            A3 = area(self._vertices[0, 0], self._vertices[0, 1],
                      self._vertices[1, 0], self._vertices[1, 1], x, y)

            # 세 부분 삼각형의 면적 합이 전체 삼각형의 면적과 같으면 내부에 있는 점
            inside[i] = abs(A - (A1 + A2 + A3)) < 1e-10

        return inside

    def make_dataset(self):
        while True:  # 충분한 포인트가 생성될 때까지 반복
            # 삼각형의 높이와 너비를 기준으로 범위 설정
            triangle_height = self._size * 1.5  # 삼각형의 높이
            triangle_width = self._size * np.sqrt(3)  # 삼각형의 너비
            
            # 먼저 충분한 수의 포인트를 생성
            n_samples_try = self._n_sample * 5  # 목표 개수의 5배로 시작 (성공 확률을 높이기 위해 증가)
            
            # 전체 영역에서 균등하게 포인트 생성
            x = np.random.uniform(-triangle_width, triangle_width, (n_samples_try, 1))
            y = np.random.uniform(-triangle_height, triangle_height, (n_samples_try, 1))
            points = np.hstack((x, y))
            
            # 각 점이 삼각형 내부에 있는지 확인
            labels = self._is_inside_triangle(points).astype(int)
            
            # 내부/외부 포인트 분리
            inside_points = points[labels == 1]
            outside_points = points[labels == 0]
            
            # 70:30 비율로 수정
            n_inside = int(self._n_sample * 0.7)  # 70%는 내부 포인트
            n_outside = self._n_sample - n_inside  # 나머지 30%는 외부 포인트
            
            # 충분한 포인트가 있는지 확인
            if len(inside_points) >= n_inside and len(outside_points) >= n_outside:
                # 내부 포인트 샘플링
                inside_indices = np.random.choice(len(inside_points), n_inside, replace=False)
                selected_inside = inside_points[inside_indices]
                
                # 외부 포인트 샘플링
                outside_indices = np.random.choice(len(outside_points), n_outside, replace=False)
                selected_outside = outside_points[outside_indices]
                
                # 최종 데이터셋 생성
                final_points = np.vstack((selected_inside, selected_outside))
                labels = np.vstack((np.ones((n_inside, 1)), np.zeros((n_outside, 1))))
                
                # 노이즈 적용
                if self._noise > 0:
                    noise_mask = np.random.random(len(labels)) < self._noise
                    labels[noise_mask] = 1 - labels[noise_mask]
                
                # x0=1인 열 추가
                x0 = np.ones((self._n_sample, 1))
                data = np.hstack((x0, final_points, labels))
                
                # 데이터 순서를 랜덤하게 섞기
                np.random.shuffle(data)
                
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
