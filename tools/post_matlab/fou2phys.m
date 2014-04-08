function res = fou2phys(u)
  ## Doubles the spectrum and inverse fft. Understand the what, not the how.
  res = real(ifftn(cat(1,u(1:end-1,:,:),conj(u(end:-1:2,:,:)))));
