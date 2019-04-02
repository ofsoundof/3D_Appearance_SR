from compute_PSNR_UP import cal_pnsr, cal_pnsr_all
import imageio
#img_sr = imageio.imread('/scratch/yawli/ColMapMiddlebury/TempleRing/x2/GeometryIterativeProcess_WeightedRegularized/iteration_0000/Pass003_0.00001/EstimatedTextures/Texture006.png')
#img_sr = imageio.imread('/scratch/yawli/ColMapMiddlebury/TempleRing/x4/GeometryIterativeProcess_WeightedRegularized/iteration_0000/Pass001_0.00100/EstimatedTextures/Texture017.png')
img_sr = imageio.imread('/scratch/yawli/ColMapMiddlebury/TempleRing/x4/GeometryIterativeProcess_WeightedRegularized/iteration_0000/Pass003_0.00001/EstimatedTextures/Texture067.png')
img_hr=imageio.imread('/scratch/yawli/ColMapMiddlebury/TempleRing/x1/WeightedBilaplacian_TV_Blender/Pass001_0.00100/EstimatedTextures/Texture000.png')

psnr_y, psnr = cal_pnsr_all(img_hr, img_sr)
print(psnr)

