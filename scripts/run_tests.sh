coverage run -m unittest discover -s tests -p '*_tests.py'
coverage report -m
coverage html
RED=$'\e[0;31m'
echo "${RED}! Make sure that .pyx files are compiled (not with pyximport), and that the compiler directives are enabled."
