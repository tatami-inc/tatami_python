<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.12.0">
  <compound kind="file">
    <name>parallelize.hpp</name>
    <path>/github/workspace/include/tatami_python/</path>
    <filename>parallelize_8hpp.html</filename>
    <namespace>tatami_python</namespace>
    <member kind="define">
      <type>#define</type>
      <name>TATAMI_PYTHON_SERIALIZE</name>
      <anchorfile>parallelize_8hpp.html</anchorfile>
      <anchor>a978af9bf0ba698bc58055a686365d4ff</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>sparse_matrix.hpp</name>
    <path>/github/workspace/include/tatami_python/</path>
    <filename>sparse__matrix_8hpp.html</filename>
    <namespace>tatami_python</namespace>
  </compound>
  <compound kind="file">
    <name>tatami_python.hpp</name>
    <path>/github/workspace/include/tatami_python/</path>
    <filename>tatami__python_8hpp.html</filename>
    <includes id="parallelize_8hpp" name="parallelize.hpp" local="yes" import="no" module="no" objc="no">parallelize.hpp</includes>
    <includes id="UnknownMatrix_8hpp" name="UnknownMatrix.hpp" local="yes" import="no" module="no" objc="no">UnknownMatrix.hpp</includes>
    <namespace>tatami_python</namespace>
  </compound>
  <compound kind="file">
    <name>UnknownMatrix.hpp</name>
    <path>/github/workspace/include/tatami_python/</path>
    <filename>UnknownMatrix_8hpp.html</filename>
    <includes id="parallelize_8hpp" name="parallelize.hpp" local="yes" import="no" module="no" objc="no">parallelize.hpp</includes>
    <class kind="struct">tatami_python::UnknownMatrixOptions</class>
    <class kind="class">tatami_python::UnknownMatrix</class>
    <namespace>tatami_python</namespace>
  </compound>
  <compound kind="class">
    <name>tatami_python::UnknownMatrix</name>
    <filename>classtatami__python_1_1UnknownMatrix.html</filename>
    <templarg>typename Value_</templarg>
    <templarg>typename Index_</templarg>
    <templarg>typename CachedValue_</templarg>
    <templarg>typename CachedIndex_</templarg>
    <base>tatami::Matrix&lt; Value_, Index_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>UnknownMatrix</name>
      <anchorfile>classtatami__python_1_1UnknownMatrix.html</anchorfile>
      <anchor>a003c8c985ee487f327250056b3f16e12</anchor>
      <arglist>(pybind11::object seed, const UnknownMatrixOptions &amp;opt)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>tatami_python::UnknownMatrixOptions</name>
    <filename>structtatami__python_1_1UnknownMatrixOptions.html</filename>
    <member kind="variable">
      <type>std::size_t</type>
      <name>maximum_cache_size</name>
      <anchorfile>structtatami__python_1_1UnknownMatrixOptions.html</anchorfile>
      <anchor>af68af40113a69a5809f4b88e32d8deeb</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>require_minimum_cache</name>
      <anchorfile>structtatami__python_1_1UnknownMatrixOptions.html</anchorfile>
      <anchor>a9785e318c954c3d2a464d10f965040f1</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>tatami_python</name>
    <filename>namespacetatami__python.html</filename>
    <class kind="class">tatami_python::UnknownMatrix</class>
    <class kind="struct">tatami_python::UnknownMatrixOptions</class>
    <member kind="function">
      <type>void</type>
      <name>parallelize</name>
      <anchorfile>namespacetatami__python.html</anchorfile>
      <anchor>a125b258daa1b77c9e7900018898ae2eb</anchor>
      <arglist>(const Function_ fun, const Index_ tasks, int threads)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>lock</name>
      <anchorfile>namespacetatami__python.html</anchorfile>
      <anchor>ac71a285d19f6a5982ff0beabc2195c8e</anchor>
      <arglist>(Function_ fun)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>parse_Sparse2darray</name>
      <anchorfile>namespacetatami__python.html</anchorfile>
      <anchor>a0d7b355c8d34f0c9363ba66a928c8859</anchor>
      <arglist>(const pybind11::object &amp;matrix, Value_ *const vbuffer, Index_ *const ibuffer, Function_ fun)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>Parse Python objects via tatami</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
