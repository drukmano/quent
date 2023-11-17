./scripts/build.sh
rm dist/quent-*.whl
python3 -m twine upload dist/quent-*.tar.gz
