

a = 0;


log_fid = fopen('timbre_matlab_log.txt', 'a');
fprintf(log_fid, strcat('[Matlab] Script starts: ', datestr(now, 'yy/mm/dd-HH:MM:SS'), '\n'));

cd 'chemin du nimp'


fprintf(log_fid, strcat('[Matlab] Script ends: ', datestr(now, 'yy/mm/dd-HH:MM:SS'), '\n'));
fclose(log_fid);