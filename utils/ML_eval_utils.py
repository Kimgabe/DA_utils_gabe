from sklearn.metrics import mean_absolute_error, mean_squared_error

# 각 모델의 이름과 해당 모델의 MSE(Mean Squared Error) 값을 저장하는 딕셔너리
my_predictions = {}

# 그래프에서 사용할 색상을 정의하는 리스트
colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive',
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray',
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
         ]


# 모듈에 포함된 모든 함수에 대한 설명 및 사용법 설명하는 함수
def describe_functions():
    descriptions = {
        'plot_predictions': """
        - 목적: 단일 모델의 예측값과 실제값을 산점도로 나타냄.
        - 사용법: plot_predictions(name_, pred, actual)
        - 매개변수:
            name_: 모델의 이름 (str)
            pred: 모델의 예측값 (list 또는 numpy array)
            actual: 모델의 실제값 (list 또는 numpy array)
        """,
        'mse_eval': """
        - 목적: 모델의 MSE를 계산하고 모델별 MSE를 시각화.
        - 사용법: mse_eval(name_, pred, actual)
        - 매개변수:
            name_: 모델의 이름 (str)
            pred: 모델의 예측값 (list 또는 numpy array)
            actual: 모델의 실제값 (list 또는 numpy array)
        """,
        'remove_model': """
        - 목적: my_predictions 딕셔너리에서 특정 모델을 제거.
        - 사용법: remove_model(name_)
        - 매개변수:
            name_: 제거할 모델의 이름 (str)
        """,
        'plot_coef': """
        - 목적: 회귀 모델의 계수를 시각화.
        - 사용법: plot_coef(columns, coef)
        - 매개변수:
            columns: 모델의 피처 이름들 (list)
            coef: 각 피처의 계수 (list 또는 numpy array)
        """,
        'add_evaluation_result': """
        - 목적: 회귀 모델의 평가 결과를 글로벌 데이터 프레임에 추가.
        - 사용법: add_evaluation_result(model_name, train_r2, validation_r2, train_mae, validation_mae, train_mse, validation_mse, train_rmse, validation_rmse)
        - 매개변수:
            model_name, train_r2, validation_r2, train_mae, validation_mae, train_mse, validation_mse, train_rmse, validation_rmse 등 평가 지표 값들
        """,
        'mse_eval_double': """
        - 목적: 두 데이터 세트에 대한 모델의 예측값과 실제값을 시각화하고 MSE 계산.
        - 사용법: mse_eval_double(name_, train_pred, train_actual, second_pred, second_actual, second_label='Validation')
        - 매개변수:
            name_, train_pred, train_actual, second_pred, second_actual, second_label
        """,
        'plot_evaluation_results': """
        - 목적: 선택된 두 평가 지표에 대한 모든 모델의 성능을 막대 그래프로 시각화.
        - 사용법: plot_evaluation_results(evaluation_metric_1, evaluation_metric_2)
        - 매개변수:
            evaluation_metric_1, evaluation_metric_2 (str)
        """
    }

    for func_name, description in descriptions.items():
        print(f"{func_name}:\n{description}\n")


"""
단일 모델의 예측값과 실제값을 산점도로 나타내는 함수
name_ 인자로 주어진 모델의 이름을 그래프 제목으로 사용하며, 예측값과 실제값을 각각 빨간색 'x' 마커와 검은색 'o' 마커로 시각화
"""
def plot_predictions(name_, pred, actual):
    # 데이터 프레임 생성
    df = pd.DataFrame({'prediction': pred, 'actual': actual})
    df = df.sort_values(by='actual').reset_index(drop=True)

    # 산점도 그리기
    plt.scatter(df.index, df['prediction'], marker='x', color='r')
    plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black')
    plt.title(name_, fontsize=15)
    plt.legend(['prediction', 'actual'], fontsize=12)


"""
모델의 예측값과 실제값을 기반으로 MSE(Mean Squared Error)를 계산해 이를 전역 딕셔너리 my_predictions에 저장
또한, 모델별 MSE를 수평 막대 그래프로 시각화
"""
def mse_eval(name_, pred, actual):
    """
    name_: 모델의 이름
    pred: 모델의 예측값
    actual: 모델의 실제값
    """
    global my_predictions  # 전역 변수로 선언된 my_predictions에 접근
    global colors  # 전역 변수로 선언된 colors에 접근

    plot_predictions(name_, pred, actual)  # 모델의 예측값과 실제값을 산점도로 시각화

    # MSE 계산 및 my_predictions에 저장
    mse = mean_squared_error(pred, actual)
    my_predictions[name_] = mse

    # my_predictions 딕셔너리를 MSE를 기준으로 정렬하여 데이터프레임 생성
    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(y_value, columns=['model', 'mse'])

    # 그래프에 사용할 범위 설정
    min_ = df['mse'].min() - 10
    max_ = df['mse'].max() + 10
    length = len(df)

    # 그래프 크기 설정
    plt.figure(figsize=(10, length))

    # 수평 막대 그래프 생성
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=15)
    bars = ax.barh(np.arange(len(df)), df['mse'])

    # 막대 그래프에 MSE 값 표시
    for i, v in enumerate(df['mse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')

    # 그래프 제목 및 범위 설정
    plt.title('MSE 에러', fontsize=18)
    plt.xlim(min_, max_)

    # 그래프 표시
    plt.show()


"""
my_predictions 딕셔너리에서 특정 모델을 제거하는 함수
모델 이름을 인자로 받아 해당 모델이 딕셔너리에 있으면 제거
함수들이 매실행마다 누적해서 그래프가 추가 되기 때문에 불필요 모델 지우기 위해 생성
"""
def remove_model(name_):
    """
    name_: 제거할 모델의 이름
    제거가 성공하면 True를 반환하고, 해당 모델이 딕셔너리에 없으면 False를 반환합니다.
    """
    global my_predictions  # 전역 변수로 선언된 my_predictions에 접근

    try:
        del my_predictions[name_]  # 해당 모델 제거 시도
    except KeyError:
        return False  # 모델이 딕셔너리에 없는 경우 False 반환
    return True  # 모델이 성공적으로 제거된 경우 True 반환



"""
회귀 모델의 계수(coefficient)를 시각화하는 함수
모델의 피처와 해당 계수를 막대 그래프로 시각화
"""
def plot_coef(columns, coef):
    """
    columns: 모델의 피처(변수) 이름들을 포함하는 리스트
    coef: 각 피처에 대한 계수(coefficient)를 포함하는 리스트
    """
    # 피처와 계수를 데이터프레임으로 변환
    coef_df = pd.DataFrame(list(zip(columns, coef)))
    coef_df.columns=['feature', 'coef']

    # 계수(coef)에 따라 내림차순으로 정렬
    coef_df = coef_df.sort_values('coef', ascending=False).reset_index(drop=True)

    # 막대 그래프 그리기
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(np.arange(len(coef_df)), coef_df['coef'])  # 수평 막대 그래프 생성
    idx = np.arange(len(coef_df))
    ax.set_yticks(idx)
    ax.set_yticklabels(coef_df['feature'])  # y축 눈금 라벨 설정
    fig.tight_layout()  # 그래프 레이아웃 조정
    plt.show()  # 그래프 표시

def add_evaluation_result(model_name, train_r2, validation_r2, train_mae, validation_mae, train_mse, validation_mse, train_rmse, validation_rmse):
    """
    회귀 모델의 평가 결과를 글로벌 데이터 프레임에 추가하는 함수.

    매개변수:
    - model_name (str): 모델의 이름.
    - train_r2 (float): 학습 데이터에 대한 R-squared 값.
    - validation_r2 (float): 검증 데이터에 대한 R-squared 값.
    - train_mae (float): 학습 데이터에 대한 Mean Absolute Error 값.
    - validation_mae (float): 검증 데이터에 대한 Mean Absolute Error 값.
    - train_mse (float): 학습 데이터에 대한 Mean Squared Error 값.
    - validation_mse (float): 검증 데이터에 대한 Mean Squared Error 값.
    - train_rmse (float): 학습 데이터에 대한 Root Mean Squared Error 값.
    - validation_rmse (float): 검증 데이터에 대한 Root Mean Squared Error 값.

    리턴값: 없음. 글로벌 데이터 프레임 `evaluation_df`에 평가 결과를 추가함.
    """
    global evaluation_df
    # 함수 내부에서 evaluation_df의 존재 여부를 확인
    try:
        evaluation_df
    except NameError:
        evaluation_df = pd.DataFrame(columns=['Model', 'Train R2', 'Validation R2', 'Train MAE', 'Validation MAE', 'Train MSE', 'Validation MSE', 'Train RMSE', 'Validation RMSE'])
    
    # 데이터 프레임에 새로운 결과 행 추가
    new_row = {
        'Model': model_name, 
        'Train R2': train_r2, 
        'Validation R2': validation_r2, 
        'Train MAE': train_mae, 
        'Validation MAE': validation_mae,
        'Train MSE': train_mse, 
        'Validation MSE': validation_mse,
        'Train RMSE': train_rmse, 
        'Validation RMSE': validation_rmse
    }
    evaluation_df = evaluation_df.append(new_row, ignore_index=True)


"""
두 개의 데이터 세트(학습 및 검증/테스트)에 대한 모델의 예측값과 실제값을 시각화하고, 각각의 MSE를 계산하는 함수
"""
def mse_eval_double(name_, train_pred, train_actual, second_pred, second_actual, second_label='Validation'):
    """
    두 개의 데이터 세트(학습 및 검증/테스트)에 대한 모델의 예측값과 실제값을 시각화하고,
    각각의 MSE, MAE, R2를 계산하여 데이터 프레임에 추가하는 함수.

    매개변수 설명:
    - name_: 모델의 이름.
    - train_pred: 학습 데이터에 대한 예측값.
    - train_actual: 학습 데이터의 실제값.
    - second_pred: 검증 또는 테스트 데이터에 대한 예측값.
    - second_actual: 검증 또는 테스트 데이터의 실제값.
    - second_label: 두 번째 데이터셋의 종류를 나타내는 문자열 (기본값은 'Validation', 'Test'로 변경 가능).
    """
    # MSE, MAE, R2 계산
    train_mse = mean_squared_error(train_actual, train_pred)
    second_mse = mean_squared_error(second_actual, second_pred)
    train_mae = mean_absolute_error(train_actual, train_pred)
    second_mae = mean_absolute_error(second_actual, second_pred)
    train_r2 = r2_score(train_actual, train_pred)
    second_r2 = r2_score(second_actual, second_pred)

    # add_evaluation_result 함수 호출하여 평가 결과 추가
    add_evaluation_result(name_, train_r2, second_r2, train_mae, second_mae, train_mse, second_mse, np.sqrt(train_mse), np.sqrt(second_mse))
    
    
    # 가로로 나란히 두 개의 산점도를 그릴 큰 캔버스 설정
    plt.figure(figsize=(24, 12))

    # 첫 번째 서브플롯: 학습 데이터에 대한 산점도
    plt.subplot(2, 2, 1)
    plot_predictions(f"{name_} - Train", train_pred, train_actual)
    plt.xlabel('Data Points')
    plt.ylabel('Predicted/Actual Value')

    # 두 번째 서브플롯: 검증/테스트 데이터에 대한 산점도
    plt.subplot(2, 2, 2)
    plot_predictions(f"{name_} - {second_label}", second_pred, second_actual)
    plt.xlabel('Data Points')
    plt.ylabel('Predicted/Actual Value')

    # MSE 계산
    train_mse = mean_squared_error(train_actual, train_pred)
    second_mse = mean_squared_error(second_actual, second_pred)

    # 색상 선택을 위한 랜덤 인덱스 생성
    color_indices = np.random.choice(len(colors), 2, replace=False)
    train_color = colors[color_indices[0]]
    second_color = colors[color_indices[1]]

    # 세 번째 서브플롯: MSE를 나타내는 수평 막대 그래프
    plt.subplot(2, 1, 2)
    plt.barh([f'{name_} - Train', f'{name_} - {second_label}'], [train_mse, second_mse], color=[train_color, second_color])
    plt.xlabel('MSE Value')
    plt.title(f"{name_} - MSE Comparison")

    # 레이아웃 조정 및 그래프 표시
    plt.tight_layout()
    plt.show()

def plot_evaluation_results(evaluation_metric_1, evaluation_metric_2):
    """
    선택된 두 평가 지표에 대한 모든 모델의 학습 및 검증 데이터 성능을 막대 그래프로 시각화하는 함수.

    매개변수:
    - evaluation_metric_1 (str): 첫 번째로 시각화할 평가 지표의 이름 (예: 'R2', 'MAE').
    - evaluation_metric_2 (str): 두 번째로 시각화할 평가 지표의 이름 (예: 'MSE', 'RMSE').

    리턴값: 없음. 선택된 평가 지표에 대한 성능 비교 막대 그래프를 표시함.
    """
    plt.figure(figsize=(18, 8))

    metrics = [evaluation_metric_1, evaluation_metric_2]
    train_metrics = [f'Train {metric}' for metric in metrics]
    validation_metrics = [f'Validation {metric}' for metric in metrics]

    x = np.arange(len(evaluation_df['Model'])) * 2  # x 값을 모델의 개수에 맞게 조정, 간격 확장

    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i+1)
        train_values = evaluation_df[f'Train {metric}']
        validation_values = evaluation_df[f'Validation {metric}']
        
        plt.bar(x - 0.4, train_values, width=0.4, label=f'Train {metric}', align='center')
        plt.bar(x, validation_values, width=0.4, label=f'Validation {metric}', align='center')
        
        plt.xlabel('Models')
