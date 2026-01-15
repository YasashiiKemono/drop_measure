# 计数模块

class DropletCounter:
    def __init__(self, line_x1, line_x2):
        self.line_x1 = line_x1
        self.line_x2 = line_x2
        self.state = None
        self.t1 = None
        self.diam_samples = []
        self.count = 0

    def update(self, cx, diameter, t):
        """
        输入当前液滴圆心 cx、直径 diameter、时间戳 t
        返回是否完成一次计数事件
        """
        event = None

        if self.state is None:
            if cx >= self.line_x1 and cx < self.line_x2:
                self.t1 = t
                self.diam_samples = [diameter]
                self.state = "waiting_second_line"
        elif self.state == "waiting_second_line":
            if cx < self.line_x2:
                self.diam_samples.append(diameter)
            else:
                # 液滴穿过第二条线，完成一次计数
                t2 = t
                avg_diameter = sum(self.diam_samples) / len(self.diam_samples)
                self.count += 1
                event = {
                    "count": self.count,
                    "t1": self.t1,
                    "t2": t2,
                    "diameter": avg_diameter
                }
                self.state = None

        return event
