def gaussMaker2d(FieldShape,GaussCenter,GaussStd): # make 2 dimensional gaussian pdf

    import numpy as np
    
    ax = np.arange(FieldShape[0])-GaussCenter[0]
    ay = np.arange(FieldShape[1])-GaussCenter[1]
    xx, yy = np.meshgrid(ax, ay)

    gauss = np.exp(-(xx**2 + yy**2) / (2. * GaussStd**2))
    pdf = gauss/np.sum(gauss)
    
    return pdf   