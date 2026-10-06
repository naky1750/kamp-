@echo off
chcp 65001 > nul
echo ============================================================
echo [Git 자동 업로드 스크립트]
echo ============================================================

set /p repo_url="GitHub 저장소 URL을 입력하세요 (예: https://github.com/username/repo.git): "

if "%repo_url%"=="" (
    echo [오류] URL이 입력되지 않았습니다.
    pause
    exit /b
)

git remote remove origin 2>nul
git remote add origin %repo_url%
git branch -M main
git add .
git commit -m "Update anomaly detection code & documentation"
git push -u origin main

echo.
echo ============================================================
echo ✓ 성공적으로 GitHub에 업로드되었습니다!
echo ============================================================
pause
