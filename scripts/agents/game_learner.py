# 사용자의 화면 스크린샷 (py auto gui? 그거)
# 게임 화면만 auto cropping
# CNN 모델 불러오기, Differentiator instantiation
# Differentiator에게 autocropped된 화면 넘기기
# 2D Grid 출력

# 위 과정을 Dynamics Handler가 수행!!

# 메인 RL (PPO) 과정

# Dynamics Handler가 Game Agent에게 2D Grid 넘기기
# Game Agent는 2D Grid input (Observation) 받고 Action 리턴
# Learner가 Action을 Dynamics Handler에게 넘기기
# Dynamics Handler가 reward, done, next observation 리턴

# Done이라면 Trajectories 완성

# Advantage 계산
# Actor Update
# Critic (Value function NN) Update
