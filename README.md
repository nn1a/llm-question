# RAG Test

## requirement

- cuda 12.1

## install python dependency

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## install ollama and pull model

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

```sh
ollama pull llama3.2
```

## run

```sh
# download tizen-docs
git clone https://github.com/samsung/tizen-docs --depth=1

# first time create embedding database with `--embeding` option
python main.py --embedding -q "GBS 로 특정 패키지 빌드 하는 방법에  대해 순서대로 알려줘"

# after create database, `--embeding` is not needed
python main.py -q "GBS 로 특정 패키지 빌드 하는 방법에  대해 순서대로 알려줘"
```

### sample result
``````
python main.py  -q "GBS 로 특정 패키지 빌드 하는 방법에  대해 순서대로 알려줘"
Loading existing vectorstore...
Loading vectorstore from: vectorstore
Creating RAG chain...
Querying RAG chain with: GBS 로 특정 패키지 빌드 하는 방법에  대해 순서대로 알려줘
Answer:
1. **Build a single package**: 
   - GBS를 사용하여 특정 패키지를 빌드할 수 있습니다.
   ```
   $ cd package_name
   $ gbs build -A architecture
   ```

2. **Build a package for different architectures**:
   - GBS는 다양한 아rchitecture를 지원합니다. 예를 들어, x86_64, i586, armv6l, armv7hl, armv7l, aarch64, mips, 및 mipsel을 지원합니다.
   ```
   $ gbs build -A armv7l #armv7l 아rchitecture를 사용하여 패키지를 빌드합니다
   $ gbs build -A i586 #i586 아rchitecture를 사용하여 패키지를 빌드합니다
   ```

3. **Make a clean build by deleting the old build root**:
   - GBS는 이전의 빌드를 삭제할 수 있습니다.
   ```
   $ cd package_name
   $ gbs --clean build -A architecture
   ```

4. **Build a package for the project**:
   - GBS를 사용하여 특정 패키지를 빌드할 수 있습니다.
   ```
   $ gbs build <gbs build option>
   ```

5. **Exclude specific packages**: 
   - GBS는 특정 패키지의 빌드를 제외할 수 있습니다.
   ```
   $ cd package_name
   $ gbs build --deps --binary-from-file package_name --exclude package_to_exclude
   ```

6. **Speed up a local build**:
   - GBS는 lokal 빌드를 빠르게 할 수 있습니다.
   ```
   $ cd package_name
   $ gbs --clean build -A architecture --speed-up
   ```

7. **Perform another build**:
   - GBS는 또 다른 빌드를 수행할 수 있습니다.
   ```
   $ cd package_name
   $ gbs build --deps --binary-from-file package_name --exclude package_to_exclude
   ```

Sources:
./tizen-docs/docs/platform/reference/gbs/gbs-build.md
./tizen-docs/docs/platform/developing/building.md
./tizen-docs/docs/platform/reference/gbs/gbs-build.md
``````