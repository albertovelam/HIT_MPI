function res = readfield(filename)
## Reads HIT in hdf5 and returns the usual spectrum
## Requires a fairly up to date version of Matlab or Octave

## Check if octave
  isOctave = exist('OCTAVE_VERSION') ~= 0;
  if isOctave
    u = load('-hdf5',filename).u;
  else
    u = hdf5read(filename,'u');
  end

  res = u(1:2:end,:,:) + j.*u(2:2:end,:,:);
