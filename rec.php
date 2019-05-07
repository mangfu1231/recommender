<!DOCTYPE html>

<html>
  <head>
    <title>Result page</title>
  </head>

  <body>
    <a href="rec.html" style="text-decoration: none"><button style="display: block;">Go Back</button></a>
  </body>
</html>

<?php 
//Form data handling
$input = $_GET["user_query"];

//Execute python script
$command = escapeshellcmd('./rec_pop.py ' . $input);
$output = shell_exec($command);
?>


<h2> <?php echo $output ?> </h2>