function dhatdz = ddz(u)

  size = size(u);
  N = size(3);

  kz = (0:N/2)'+0*j;
  kz = reshape(kz,N/2+1,1,1);
  dhatdz = zeros(N/2+1,N,N);

  for i = 1:N
    for j = 1:N
      dhatdz(:,j,i) = 1*j*kz.*u(:,j,i);
    end
  end
