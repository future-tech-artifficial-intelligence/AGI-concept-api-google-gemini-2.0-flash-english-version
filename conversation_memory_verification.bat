@echo off
echo Verifying conversation coherence...
echo.

echo Executing memory test...
python -m tests.test_memory_retrieval_enhancer

echo.
echo Test finished. Check the results above.
pause
