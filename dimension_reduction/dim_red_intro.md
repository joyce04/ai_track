## Dimension Reduction

### Definition

- Similar to compressing the data:  **Compress large set of features onto a new feature subspace of lower dimensional without losing the important information.**

- Dimension reduction will lose some information

- 낮은 차원으로 데이터를 매핑하는 방식을 찾아내는 것. Finding a smaller set of new variables, each being a combination of the original input. 데이터의 내재적 속성을 찾아내는 것.


### Why important?

- Solve the curse of dimensionality - 차원이 높아질 수록 Euclidean distance가 예상치 못한 방식으로 동작한다.
- Can reduce the complexity required to train machine learning models
- Can eliminate overfitting
- Remove redundant features (multicollinearity)
- Visualizing data

### Techniques

- Supervised :

  - Linear mapping - Linear Discriminant Analysis (LDA): 클래스를 최대한 잘 분리하는 projection을 찾는다. 같은 클래스의 variance는 줄이고, 다른 클래스간의 variance를 키우는 방향으로 움직인다.
    ![image-20181228113410018](../resources/image-20181228113410018.png)

    ``` python
    from sklearn.lda import LDA
    LDA(n_components=3).fit_transform(X_train, y_train)
    ```

  - CNN Pooling

- Unsupervised : 

  - Linear mapping - Principal Component Analysis (PCA): find points with maximum variance
    ``` python
    from sklearn.decomposition imiport PCA
    PCA(n_components=3).fit_transform(X_train)
    ```

    Principal Component : 데이터를 가장 넓게 펼치는 축 = 가장 큰 분산을 가지는 축

    Covariance(공분산) :점들의 그룹과 다른 점들의 그룹과의 관계를 나타내는 지표.

  - Autoencoder 

### Applications



References :

- https://medium.com/fintechexplained/what-is-dimension-reduction-in-data-science-2aa5547f4d29
- http://sanghyukchun.github.io/72/
- https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/