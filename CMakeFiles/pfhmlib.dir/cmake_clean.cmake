FILE(REMOVE_RECURSE
  "lib/libpfhmlib.pdb"
  "lib/libpfhmlib.so"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/pfhmlib.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
