import re

# 입력 텍스트
text = """
Intersection between triangle 34 and triangle 39. Coplanar: No
Intersection between triangle 35 and triangle 32. Coplanar: No
Intersection between triangle 36 and triangle 32. Coplanar: No
Intersection between triangle 37 and triangle 34. Coplanar: No
Intersection between triangle 38 and triangle 34. Coplanar: No
Intersection between triangle 39 and triangle 34. Coplanar: No
Intersection between triangle 40 and triangle 34. Coplanar: No
Intersection between triangle 43 and triangle 41. Coplanar: No
Intersection between triangle 32 and triangle 35. Coplanar: No
Intersection between triangle 34 and triangle 40. Coplanar: No
Intersection between triangle 36 and triangle 33. Coplanar: No
Intersection between triangle 41 and triangle 43. Coplanar: No
Intersection between triangle 42 and triangle 43. Coplanar: No
Intersection between triangle 43 and triangle 42. Coplanar: No
Intersection between triangle 0 and triangle 7. Coplanar: No
Intersection between triangle 1 and triangle 2. Coplanar: No
Intersection between triangle 2 and triangle 1. Coplanar: No
Intersection between triangle 3 and triangle 5. Coplanar: No
Intersection between triangle 4 and triangle 7. Coplanar: No
Intersection between triangle 5 and triangle 3. Coplanar: No
Intersection between triangle 6 and triangle 3. Coplanar: No
Intersection between triangle 7 and triangle 4. Coplanar: No
Intersection between triangle 8 and triangle 7. Coplanar: No
Intersection between triangle 12 and triangle 10. Coplanar: No
Intersection between triangle 13 and triangle 10. Coplanar: No
Intersection between triangle 16 and triangle 20. Coplanar: No
Intersection between triangle 17 and triangle 26. Coplanar: No
Intersection between triangle 18 and triangle 21. Coplanar: No
Intersection between triangle 19 and triangle 21. Coplanar: No
Intersection between triangle 20 and triangle 16. Coplanar: No
Intersection between triangle 21 and triangle 18. Coplanar: No
Intersection between triangle 23 and triangle 25. Coplanar: No
Intersection between triangle 24 and triangle 25. Coplanar: No
Intersection between triangle 25 and triangle 24. Coplanar: No
Intersection between triangle 26 and triangle 17. Coplanar: No
Intersection between triangle 27 and triangle 28. Coplanar: No
Intersection between triangle 28 and triangle 27. Coplanar: No
Intersection between triangle 31 and triangle 29. Coplanar: No
Intersection between triangle 32 and triangle 36. Coplanar: No
Intersection between triangle 33 and triangle 36. Coplanar: No
Intersection between triangle 34 and triangle 38. Coplanar: No
Intersection between triangle 34 and triangle 37. Coplanar: No
Intersection between triangle 3 and triangle 6. Coplanar: No
Intersection between triangle 7 and triangle 0. Coplanar: No
Intersection between triangle 9 and triangle 4. Coplanar: No
Intersection between triangle 10 and triangle 12. Coplanar: No
Intersection between triangle 11 and triangle 12. Coplanar: No
Intersection between triangle 12 and triangle 11. Coplanar: No
Intersection between triangle 21 and triangle 19. Coplanar: No
Intersection between triangle 22 and triangle 16. Coplanar: No
Intersection between triangle 25 and triangle 23. Coplanar: No
Intersection between triangle 29 and triangle 31. Coplanar: No
Intersection between triangle 7 and triangle 8. Coplanar: No
Intersection between triangle 9 and triangle 0. Coplanar: No
Intersection between triangle 10 and triangle 13. Coplanar: No
Intersection between triangle 14 and triangle 31. Coplanar: No
Intersection between triangle 15 and triangle 22. Coplanar: No
Intersection between triangle 16 and triangle 22. Coplanar: No
Intersection between triangle 22 and triangle 15. Coplanar: No
Intersection between triangle 31 and triangle 14. Coplanar: No
Intersection between triangle 0 and triangle 9. Coplanar: No
Intersection between triangle 4 and triangle 9. Coplanar: No
Intersection between triangle 8 and triangle 9. Coplanar: No
Intersection between triangle 9 and triangle 8. Coplanar: No
"""

text2 = """
Intersection between triangle 155011 and triangle 155047.
Intersection between triangle 229252 and triangle 229297.
Intersection between triangle 229263 and triangle 229198.
Intersection between triangle 229297 and triangle 229198.
Intersection between triangle 155047 and triangle 155011.
Intersection between triangle 274022 and triangle 274014.
Intersection between triangle 274014 and triangle 274022.
Intersection between triangle 274015 and triangle 274022.
Intersection between triangle 263323 and triangle 259426.
Intersection between triangle 274015 and triangle 274032.
Intersection between triangle 263323 and triangle 259425.
Intersection between triangle 274032 and triangle 274015.
Intersection between triangle 274022 and triangle 274015.
Intersection between triangle 282283 and triangle 490124.
Intersection between triangle 270521 and triangle 477343.
Intersection between triangle 320869 and triangle 350351.
Intersection between triangle 302389 and triangle 302383.
Intersection between triangle 302389 and triangle 302366.
Intersection between triangle 320872 and triangle 350351.
Intersection between triangle 350351 and triangle 320872.
Intersection between triangle 317359 and triangle 317298.
Intersection between triangle 357130 and triangle 357209.
Intersection between triangle 302383 and triangle 302389.
Intersection between triangle 483470 and triangle 483458.
Intersection between triangle 453746 and triangle 453751.
Intersection between triangle 483449 and triangle 483470.
Intersection between triangle 471070 and triangle 471065.
Intersection between triangle 456757 and triangle 456631.
Intersection between triangle 456760 and triangle 456631.
Intersection between triangle 477339 and triangle 267529.
Intersection between triangle 477339 and triangle 267530.
Intersection between triangle 483470 and triangle 483449.
Intersection between triangle 536286 and triangle 684436.
Intersection between triangle 541322 and triangle 541367.
Intersection between triangle 502079 and triangle 502083.
Intersection between triangle 469816 and triangle 471406.
Intersection between triangle 424385 and triangle 424381.
Intersection between triangle 536290 and triangle 555436.
Intersection between triangle 502083 and triangle 502079.
Intersection between triangle 418561 and triangle 418578.
Intersection between triangle 418562 and triangle 418578.
Intersection between triangle 502079 and triangle 502100.
Intersection between triangle 508888 and triangle 508886.
Intersection between triangle 482123 and triangle 482116.
Intersection between triangle 579212 and triangle 579217.
Intersection between triangle 579217 and triangle 579212.
Intersection between triangle 544686 and triangle 544705.
Intersection between triangle 572924 and triangle 688063.
Intersection between triangle 654519 and triangle 679664.
Intersection between triangle 555436 and triangle 684436.
Intersection between triangle 555436 and triangle 536290.
Intersection between triangle 684102 and triangle 684115.
Intersection between triangle 684115 and triangle 684102.
Intersection between triangle 687002 and triangle 688063.
Intersection between triangle 754720 and triangle 701562.
Intersection between triangle 759948 and triangle 759759.
Intersection between triangle 683485 and triangle 683508.
Intersection between triangle 683485 and triangle 683513.
Intersection between triangle 684436 and triangle 536286.
Intersection between triangle 683508 and triangle 683485.
Intersection between triangle 691063 and triangle 691056.
Intersection between triangle 684436 and triangle 555436.
Intersection between triangle 691056 and triangle 691063.
Intersection between triangle 683513 and triangle 683485.
Intersection between triangle 691057 and triangle 691063.
"""


# 삼각형 번호 추출
triangles = re.findall(r'triangle (\d+)', text)
triangles2 = re.findall(r'triangle (\d+)', text2)

# 중복 제거
unique_triangles = set(triangles)
unique_triangles2 = set(triangles2)

# 중복 제거된 삼각형의 개수 출력
print(len(unique_triangles))


# 숫자 열 그룹을 리스트로 표현

if unique_triangles == unique_triangles2:
    print("Same")
else:
    print("Different")