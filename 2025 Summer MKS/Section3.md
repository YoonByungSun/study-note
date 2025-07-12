# Section3. Unity ML-Agents 살펴보기
***
[Week2 실습](https://github.com/YoonByungSun/study-note/tree/main/2025%20Summer%20MKS/Week2)  
***

## 1. Unity ML-Agents란?  

- Unity를 이용한 인공지능 Agent 학습을 지원하는 오픈소스  
- 기본적으로 강화학습을 위한 다양한 기능 제공  
- Agent가 다양한 경험을 수행하며 학습  
- 실제 환경 적용이 어려운 강화학습의 단점 해결  
  - 시공간과 실패에 제약이 없음  

#### - 적용과정  
1. Unity를 이용한 환경 제작  
2. 제작된 환경에 ML Agent 적용 및 설정  
3. 강화학습을 통해 Agent 학습  
4. 학습이 완료된 Agent 모델을 다시 Unity에 임베딩  
5. Unity 환경을 빌드하여 학습된 Agent를 적용  

***

## 2. Unity 개발 환경 구성

Unity 6000.0.41f1  
Unity ML-Agents 3.0.0  
Python 3.10  

***

## 3. ML-Agents 살펴보기  

- ### Agent Inspector  
  - #### Behavior Parameters (필수 - 학습 관련)
    - Behavior Name
    - Vector Observation
    - Actions
    - Model
    - Behavior Type
    - Team Id
    - Use Child Sensors
    - Observable Attribute Handling
  - #### Ball 3D Agent (필수 - Agent 관련 스크립트)
    - Max Step
    - Specific to Ball3D

- ### Ball 3D Agent (Script)
  - #### Initialize()  
    환경이 실행될 때 호출되는 초기화 함수
    ```cs
    public override void Initialize()
    {
      m_BallRb = ball.GetComponent<RigidBody>();
      m_ResetParams = Academy.Instance.EnvironmentParameters;
      SetResetParameters();
    }
    ```

  - #### CollectObservations(VectorSensor)  
    Agent에게 Vector Observation 정보를 전달해주는 함수
    ```cs
    public override void CollectObservations(VectorSensor sensor)
    {
      if (useVecObs)
      {
        sensor.AddObservation(gameObject.transform.rotation.z);  // z rotation 값 (scalar)
        sensor.AddObservation(gameObject.transform.rotation.x);  // x rotation 값 (scalar)
        sensor.AddObservation(ball.transform.position - gameObject.transform.position);  // ball과의 거리 (vector)
        sensor.AddObservation(m_BallRb.velocity);  // ball의 속도 (vector)
      }
    }
    ```

  - #### OnActionReceived(ActionBuffers)  
    Agent가 결정한 행동을 전달, 보상 업데이트, 에피소드 종료
    ```cs
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
      // Agent가 결정한 행동을 Action Buffer에서 가져옴
      var actionZ = 2f * Mathf.Clamp(actionBuffers.Continuous[0], -1f, 1f);
      var actionX = 2f * Mathf.Clamp(actionBuffers.Continuous[1], -1f, 1f);

      // 가져온 행동을 환경에 반영
      if ((gameObject.transform.rotation.z < 0.25f && actionZ > 0f) ||
          (gameObject.transform.rotation.z > -0.25f && actionZ < 0f))
      {
        gameObject.transfrom.Rotate(new Vector3(0, 0, 1), actionZ);
      }

      if ((gameObject.transform.rotation.x < 0.25f && actionX > 0f) ||
          (gameObject.transform.rotation.x > -0.25f && actionX < 0f))
      {
        gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
      }

      // 보상과 에피소드 종료 or 패널티를 결정
      // 공이 떨어지는 상황에서, 공과 판 사이의 거리를 기준으로 보상 or 패널티를 결정하는 예시
      if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
          Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
          Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
      {
        SetReward(-1f);
        EndEpisode();
      }
      else
      {
        SetReward(0.1f);
      }
    }
    ```

  - #### OnEpisodeBegin()  
    각 에피소드가 시작될 때 호출되는 함수
    ```cs
    public override void OnEpisodeBegin()
    {
      gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
      gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
      gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));
      m_BallRb.velocity = new Vector3(0f, 0f, 0f);
      ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f))
        + gameObject.transform.position;
      // Reset the parameters when the Agent is reset.
      SetResetParemeters();
    }
    ```

  - #### Heuristic(in ActionBuffers)  
    개발자가 직접 명령을 내리는 휴리스틱 모드에서 사용 (주로 테스트, 모방 학습에 사용)
    ```cs
    public override void Heuristic(in ActionBuffers actionsOut)
    {
      var continuousActionsOut = actionsOut.ContinuousActions;
      continuousActionsOut[0] = -Input.GetAxis("Horizontal");
      continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
    ```

- ### Decision Requester  
  Agent의 행동을 정책에게 요청하는 컴포넌트
- ### Model Overrider (Script)  
  학습이 완료된 후 모델의 유효성을 검사하기 위해 내부적으로 사용되는 클래스 (ML Agents 설정에 필수 요소는 아님)