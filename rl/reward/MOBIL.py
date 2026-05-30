import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math

# ==========================================
# 設定パラメータ
# ==========================================
SIMULATION_STEPS = 1000   # フレーム数
DT = 0.5                 # 時間刻み幅

# IDM (追従モデル) パラメータ
IDM_PARAMS = {
    'v0': 0.8,    # 希望速度
    'T': 1.5,     # 車間時間
    'a': 0.3,     # 最大加速度
    'b': 0.3,     # 快適減速度
    's0': 2.0,    # 最小車間距離
    'delta': 4.0  # 指数
}

# MOBIL (車線変更モデル) パラメータ
MOBIL_PARAMS = {
    'p': 0.2,           # 礼儀正しさ: 0=自己中, 1=利他
    'b_safe': 0.5,      # 安全ブレーキ限界
    'a_thr': 0.1        # 変更閾値
}

STOP_LINE = 15.0         # 停止線位置
SPAWN_RATE = 0.15        # 車両発生率
CAR_LENGTH = 2.0         # 車両長
LANE_WIDTH = 4.0         # 車線幅

# 信号機の設定
GREEN_DURATION = 100
YELLOW_DURATION = 65

# ==========================================
# 物理計算関数 (IDM)
# ==========================================

def calculate_idm_accel(v, v_leader, gap, params):
    """ IDMによる加速度計算 """
    v0 = params['v0']
    T = params['T']
    a = params['a']
    b = params['b']
    s0 = params['s0']
    delta = params['delta']

    delta_v = v - v_leader
    s_star = s0 + v * T + (v * delta_v) / (2 * math.sqrt(a * b))

    if gap <= 0.1: gap = 0.1

    acc = a * (1 - (v / v0)**delta - (s_star / gap)**2)
    return acc

# ==========================================
# クラス定義
# ==========================================

class Car:
    def __init__(self, x, y, dx, dy, direction, lane_id, params):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        # 速度に少しバラつきを持たせて追い越しの動機を作る
        self.speed = params['v0'] * random.uniform(0.7, 1.0)
        self.direction = direction
        self.lane_id = lane_id  # 0:内側, 1:外側
        self.params = params
        self.color = 'cyan'     # 通常色
        self.change_cooldown = 0 # 車線変更直後のクールダウン

    def get_neighbors(self, cars, check_lane_offset=0):
        """
        指定した車線（現在の車線 + check_lane_offset）における
        前走車(leader)と後続車(follower)を探す
        """
        target_lane_id = self.lane_id + check_lane_offset
        if target_lane_id < 0 or target_lane_id > 1:
            return None, None

        leader = None
        follower = None
        min_dist_leader = 10000.0
        min_dist_follower = 10000.0

        # 車線幅の判定用座標
        if self.direction == 'H':
            if self.dx > 0: target_y = -2 - (target_lane_id * LANE_WIDTH)
            else:           target_y = 2 + (target_lane_id * LANE_WIDTH)
            target_x = self.x 
        else:
            if self.dy > 0: target_x = 2 + (target_lane_id * LANE_WIDTH)
            else:           target_x = -2 - (target_lane_id * LANE_WIDTH)
            target_y = self.y

        for other in cars:
            if other is self: continue

            # 同じ方向、かつ指定した車線（座標）にいるか判定
            is_same_lane = False
            if self.direction == 'H' and other.direction == 'H':
                if abs(other.y - target_y) < 0.5: is_same_lane = True
            elif self.direction == 'V' and other.direction == 'V':
                if abs(other.x - target_x) < 0.5: is_same_lane = True

            if not is_same_lane: continue

            # 前後の判定
            rel_dist = 0
            if self.direction == 'H':
                rel_dist = (other.x - self.x) * self.dx
            else:
                rel_dist = (other.y - self.y) * self.dy

            if rel_dist > 0: # 前方
                if rel_dist < min_dist_leader:
                    min_dist_leader = rel_dist
                    leader = other
            else: # 後方
                if abs(rel_dist) < min_dist_follower:
                    min_dist_follower = abs(rel_dist)
                    follower = other

        return leader, follower

    def check_lane_change(self, cars, current_acc):
        """ MOBILモデルによる車線変更判断 """
        if self.change_cooldown > 0:
            self.change_cooldown -= 1
            return False

        # 交差点付近（停止線の内側、または手前すぎる場合）は変更禁止
        pos = self.x if self.direction == 'H' else self.y
        if abs(pos) < STOP_LINE + 5.0:
            return False

        target_offset = 1 if self.lane_id == 0 else -1
        new_leader, new_follower = self.get_neighbors(cars, check_lane_offset=target_offset)

        # 1. 安全基準
        acc_new_follower_tilde = 0.0
        if new_follower:
            gap = 0
            if self.direction == 'H': gap = abs(self.x - new_follower.x) - CAR_LENGTH
            else:                     gap = abs(self.y - new_follower.y) - CAR_LENGTH

            if gap < CAR_LENGTH: return False # 衝突するなら不可
            acc_new_follower_tilde = calculate_idm_accel(new_follower.speed, self.speed, gap, new_follower.params)

            if acc_new_follower_tilde < -MOBIL_PARAMS['b_safe']:
                return False

        # 2. 動機基準
        gap_to_new_leader = 10000.0
        v_new_leader = 0.0
        if new_leader:
            if self.direction == 'H': gap_to_new_leader = abs(new_leader.x - self.x) - CAR_LENGTH
            else:                     gap_to_new_leader = abs(new_leader.y - self.y) - CAR_LENGTH
            v_new_leader = new_leader.speed

        acc_self_tilde = calculate_idm_accel(self.speed, v_new_leader, gap_to_new_leader, self.params)

        acc_new_follower = 0.0
        if new_follower:
            gap_old = 10000.0
            v_old_leader = 0.0
            if new_leader: # ここでは簡易的にnew_leaderとの距離を使う（本来はold_leader）
                 if self.direction == 'H': gap_old = abs(new_leader.x - new_follower.x) - CAR_LENGTH
                 else:                     gap_old = abs(new_leader.y - new_follower.y) - CAR_LENGTH
                 v_old_leader = new_leader.speed
            acc_new_follower = calculate_idm_accel(new_follower.speed, v_old_leader, gap_old, new_follower.params)

        my_advantage = acc_self_tilde - current_acc
        others_disadvantage = acc_new_follower_tilde - acc_new_follower

        total_incentive = my_advantage + MOBIL_PARAMS['p'] * others_disadvantage

        if total_incentive > MOBIL_PARAMS['a_thr']:
            return True

        return False

    def change_lane(self):
        """ 実際に座標を変更する """
        offset_dir = 1 if self.lane_id == 0 else -1
        self.lane_id += offset_dir

        if self.direction == 'H':
            if self.dx > 0: self.y -= offset_dir * LANE_WIDTH 
            else:           self.y += offset_dir * LANE_WIDTH
        else:
            if self.dy > 0: self.x += offset_dir * LANE_WIDTH
            else:           self.x -= offset_dir * LANE_WIDTH

        self.change_cooldown = 20
        self.color = 'magenta' # 車線変更のエフェクト

    def update(self, cars, traffic_light_state, dt):
        # 1. 前走車の探索
        leader, _ = self.get_neighbors(cars, check_lane_offset=0)

        min_gap = 10000.0
        leader_speed = 0.0
        if leader:
            if self.direction == 'H': min_gap = abs(leader.x - self.x) - CAR_LENGTH
            else:                     min_gap = abs(leader.y - self.y) - CAR_LENGTH
            leader_speed = leader.speed

        # 2. 信号（停止線）の考慮
        dist_to_stop = 10000.0
        must_stop = False

        # 信号判定ロジック
        if self.direction == 'H' and traffic_light_state in ['V_GREEN', 'V_YELLOW']:
            if self.dx > 0 and self.x < -STOP_LINE: 
                dist_to_stop = -STOP_LINE - self.x
                must_stop = True
            elif self.dx < 0 and self.x > STOP_LINE: 
                dist_to_stop = self.x - STOP_LINE
                must_stop = True

        if self.direction == 'V' and traffic_light_state in ['H_GREEN', 'H_YELLOW']:
            if self.dy > 0 and self.y < -STOP_LINE:
                dist_to_stop = -STOP_LINE - self.y
                must_stop = True
            elif self.dy < 0 and self.y > STOP_LINE:
                dist_to_stop = self.y - STOP_LINE
                must_stop = True

        # 停止線を「速度0の前走車」として扱う
        if must_stop and dist_to_stop < min_gap:
            min_gap = dist_to_stop
            leader_speed = 0.0

        # 3. 加速度計算 & 車線変更判断
        curr_acc = calculate_idm_accel(self.speed, leader_speed, min_gap, self.params)

        if not must_stop:
            if self.check_lane_change(cars, curr_acc):
                self.change_lane()

        # 4. 物理更新
        self.speed += curr_acc * dt
        if self.speed < 0: self.speed = 0

        self.x += self.dx * self.speed * dt
        self.y += self.dy * self.speed * dt

        if self.change_cooldown < 15 and self.color == 'magenta':
            self.color = 'cyan'

# ==========================================
# シミュレーション管理と描画
# ==========================================

class TrafficSimulation:
    def __init__(self):
        self.cars = []
        self.time = 0
        self.light_state = 'H_GREEN'
        self.light_timer = 0

    def step(self):
        self.time += 1
        self.update_traffic_lights()
        self.spawn_cars()
        for car in self.cars:
            car.update(self.cars, self.light_state, DT)
        self.cars = [c for c in self.cars if -60 < c.x < 60 and -60 < c.y < 60]

    def update_traffic_lights(self):
        self.light_timer += 1
        if self.light_state == 'H_GREEN' and self.light_timer > GREEN_DURATION:
            self.light_state = 'H_YELLOW'
            self.light_timer = 0
        elif self.light_state == 'H_YELLOW' and self.light_timer > YELLOW_DURATION:
            self.light_state = 'V_GREEN'
            self.light_timer = 0
        elif self.light_state == 'V_GREEN' and self.light_timer > GREEN_DURATION:
            self.light_state = 'V_YELLOW'
            self.light_timer = 0
        elif self.light_state == 'V_YELLOW' and self.light_timer > YELLOW_DURATION:
            self.light_state = 'H_GREEN'
            self.light_timer = 0

    def spawn_cars(self):
        if random.random() < SPAWN_RATE:
            route = random.choice(['W-E', 'E-W', 'S-N', 'N-S'])
            lane = random.randint(0, 1) # ランダムな車線に出現

            # 出現座標の計算
            if route == 'W-E':
                y_pos = -2 - (lane * LANE_WIDTH)
                spawn_car = Car(-55, y_pos, 1, 0, 'H', lane, IDM_PARAMS)
            elif route == 'E-W':
                y_pos = 2 + (lane * LANE_WIDTH)
                spawn_car = Car(55, y_pos, -1, 0, 'H', lane, IDM_PARAMS)
            elif route == 'S-N':
                x_pos = 2 + (lane * LANE_WIDTH)
                spawn_car = Car(x_pos, -55, 0, 1, 'V', lane, IDM_PARAMS)
            elif route == 'N-S':
                x_pos = -2 - (lane * LANE_WIDTH)
                spawn_car = Car(x_pos, 55, 0, -1, 'V', lane, IDM_PARAMS)

            # 重なり防止チェック
            is_safe = True
            for c in self.cars:
                dist_sq = (c.x - spawn_car.x)**2 + (c.y - spawn_car.y)**2
                if dist_sq < (IDM_PARAMS['s0'] * 3)**2:
                    is_safe = False
                    break
            if is_safe:
                self.cars.append(spawn_car)

sim = TrafficSimulation()
fig, ax = plt.subplots(figsize=(8, 8))

def draw_background():
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_facecolor('#333333')

    # 道路描画
    road_w = 16 
    ax.add_patch(plt.Rectangle((-50, -road_w/2), 100, road_w, color='#555555'))
    ax.add_patch(plt.Rectangle((-road_w/2, -50), road_w, 100, color='#555555'))

    # ライン描画
    ax.plot([-50, 50], [0, 0], color='yellow', linestyle='-', linewidth=2)
    ax.plot([0, 0], [-50, 50], color='yellow', linestyle='-', linewidth=2)
    ax.plot([-50, 50], [-4, -4], color='white', linestyle='--', linewidth=1)
    ax.plot([-50, 50], [4, 4], color='white', linestyle='--', linewidth=1)
    ax.plot([-4, -4], [-50, 50], color='white', linestyle='--', linewidth=1)
    ax.plot([4, 4], [-50, 50], color='white', linestyle='--', linewidth=1)

    # 停止線
    for d in [-1, 1]:
        ax.plot([-road_w/2, 0], [d*STOP_LINE, d*STOP_LINE], color='white', linewidth=3)
        ax.plot([0, road_w/2], [d*STOP_LINE, d*STOP_LINE], color='white', linewidth=3)
        ax.plot([d*STOP_LINE, d*STOP_LINE], [-road_w/2, 0], color='white', linewidth=3)
        ax.plot([d*STOP_LINE, d*STOP_LINE], [0, road_w/2], color='white', linewidth=3)

def init(): return []

def animate(i):
    ax.clear()
    draw_background()
    sim.step()

    for c in sim.cars:
        ax.plot(c.x, c.y, 's', color=c.color, markersize=8, markeredgecolor='black')

    light_h = '#00FF00' if 'GREEN' in sim.light_state else ('#FFFF00' if 'YELLOW' in sim.light_state else '#FF0000')
    light_v = '#FF0000' if 'GREEN' in sim.light_state else ('#FF0000' if 'YELLOW' in sim.light_state else '#00FF00' if 'V_GREEN' in sim.light_state else '#FFFF00')

    ax.text(-45, 45, f"EW: ●", color=light_h, fontsize=20, fontweight='bold')
    ax.text(35, 45, f"NS: ●", color=light_v, fontsize=20, fontweight='bold')
    ax.set_title(f"Time: {i}")
    ax.set_xticks([]); ax.set_yticks([])

ani = animation.FuncAnimation(fig, animate, frames=SIMULATION_STEPS, init_func=init, interval=50)

print("動画生成を開始します...")
try:
    ani.save('traffic_sim_mobil.mp4', writer='ffmpeg', fps=20)
    print("保存完了: traffic_sim_mobil.mp4")
except Exception as e:
    print(f"MP4保存失敗（ffmpeg未検出など）: {e}")
    try:
        ani.save('traffic_sim_mobil.gif', writer='pillow', fps=20)
        print("保存完了: traffic_sim_mobil.gif")
    except Exception as e2:
        print(f"GIF保存失敗: {e2}")

plt.close()