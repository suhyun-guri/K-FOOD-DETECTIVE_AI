# K-FOOD-DETECTIVE 🍱
K-FOOD DETECTIVE : Find out the Korean foods that you saw for the first time in your favorite korean movies or dramas

## 📌 프로젝트 주제
한류에 관심이 많은 외국인 사용자가 한국 매체(영화, 드라마 등)에서 본 한식이 무엇인지 알려주는 AI image detection 서비스

### 👥 팀 구성원의 전체 이름과 역할

| 이름 | 역할 |
|----|----|
| 박서린 | FE |
| 이무원 | BE |
| 최성민 | BE |
| 김수현 | AI |
| 이서윤 | AI |
<hr>


## 📌 서비스 설명
### **1. 기획 의도 및 기대 효과** <br>
오징어게임, 기생충과 같은 영화, 드라마 등 K-미디어콘텐츠의 세계적 유행을 통해 그 안에 등장하는 한식 역시 자연스레 더 많은 외국인들에게 노출되고 있습니다.
한식을 궁금해하는 외국인들을 위해 음식 사진을 업로드하면 해당 음식이 무엇인지 알려주는 한식 기반의 AI image detection 서비스를 기획하였습니다.

추가적으로 해당 음식이 사용자의 입맛과 얼마나 일치하는지 알 수 있는 테스트와 직접 만들어 볼 수 있는 레시피를 제공하여 한식에 대한 외국인 사용자의 궁금증과 니즈를 최대한 해결할 수 있도록 서비스를 구성하였습니다.
해당 서비스를 통해 외국인의 한식에 대한 접근성을 높이고 허들을 낮추어, 한식에 대한 해외에서의 관심이 더 증대하기를 기대합니다.
### **2. 웹 서비스의 최종적인 메인기능과 서브기능 (예: 시스템의 입출력 동작과 범위 설명)**  <br>

`메인 기능`
- 한식이 나오는 이미지파일이나 그 url을 업로드하면 detection을 거쳐 해당 이미지 속 음식이 무엇인지를 간단한 정보와 함께 제공하는 기능
- 사용자의 음식 취향에 대한 정보를 받아 detection의 결과로서 나온 음식과의 적합도를 알려주는 기능
- 결과로 나온 음식을 직접 요리해보고 싶은 사용자를 위해 버튼클릭을 통해 해당 음식에 대한 영문 레시피 사이트로 연결하는 기능

`서브 기능`
- 회원가입 및 로그인 기능
- 음식 정보 및 레시피와 음식 적합도 결과 북마킹 기능
- 음식 별 정보 공유 및 소통을 위한 댓글 등록 및 수정/삭제 기능
### 스토리보드 & 시나리오

- [전체 페이지 - Figma](https://www.figma.com/file/QPlgmWw0P9GKdL4x3zItNK/%5BCompleted%5D-K-Food-Searching?node-id=0%3A1) <br>
- [구현 예시](https://www.figma.com/proto/QPlgmWw0P9GKdL4x3zItNK/%5BCompleted%5D-K-Food-Searching?node-id=4%3A2&scaling=scale-down&page-id=0%3A1&starting-point-node-id=4%3A2)

`YOLOv5 모델의 학습 결과 및 Flow`
<p align='left'>
  <img src=https://user-images.githubusercontent.com/70987343/158050118-5447786a-86c7-4551-aac2-6416d86236bf.png width=800px float=left>
</p>

`추천 시스템의 Flow`
<p align='left'>
  <img src=https://user-images.githubusercontent.com/70987343/158050125-ea5cc6ca-91bc-4ee6-82e2-e99b7733c398.png width=800px>
</p>
