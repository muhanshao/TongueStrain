#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nibabel as nib
import numpy as np
from dipy.sims.voxel import sticks_and_ball
from dipy.sims.voxel import multi_tensor
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import (sph_harm_ind_list,
                              real_sym_sh_mrtrix,
                              lazy_index)
from dipy.core.geometry import cart2sphere
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh
import pdb


class BasicDWI():
    def __init__(self, fbvals, fbvecs, fmask):
        # self.fbvals = fbvals
        # self.fbvecs = fbvecs
        # self.fmask = fmask

        self.bvals, self.bvecs = read_bvals_bvecs(fbvals, fbvecs)
        self.gtab = gradient_table(self.bvals, self.bvecs)

        mask = nib.load(fmask)
        self.affine = mask.affine
        self.header = mask.header
        self.mask = mask.get_data()
        self.header.set_data_dtype(np.float32)

    def gen_syn_dwi(self, fdir1, fdir2, ffrac1, ffrac2,
                    fdiffu, fS0, snr, fdwi):
        dir1 = nib.load(fdir1).get_data()
        dir2 = nib.load(fdir2).get_data()
        frac1 = nib.load(ffrac1).get_data()
        frac2 = nib.load(ffrac2).get_data()
        diffu = nib.load(fdiffu).get_data()
        S0 = nib.load(fS0).get_data()
        dwi = np.zeros(list(self.mask.shape) + [len(self.bvals)])

        x, y, z = (self.mask > 0).nonzero()
        frac1 = frac1 * 100
        frac2 = frac2 * 100
        for pos in range(len(x)):
            i, j, k = x[pos], y[pos], z[pos]
            d = diffu[i, j, k]
            s0 = S0[i, j, k]
            if frac2[i, j, k] > 0.:
                fractions = [frac1[i, j, k], frac2[i, j, k]]
                angles = [dir1[i, j, k, :], dir2[i, j, k, :]]
            else:
                fractions = [frac1[i, j, k]]
                angles = [dir1[i, j, k, :]]
            dwi[i, j, k, :], _ = sticks_and_ball(self.gtab, d=d, S0=s0,
                                                 angles=angles,
                                                 fractions=fractions,
                                                 snr=snr)

        out = nib.Nifti1Image(dwi, self.affine, self.header)
        out.to_filename(fdwi)

    def gen_syn_multi_tensor(self, fdir1, fdir2, snr, fdwi,
                             frac1=70):
        dir1 = nib.load(fdir1).get_data()
        dir2 = nib.load(fdir2).get_data()

        S0 = 1.
        dwi = np.zeros(list(self.mask.shape) + [len(self.bvals)])

        x, y, z = (self.mask > 0).nonzero()

        for pos in range(len(x)):
            i, j, k = x[pos], y[pos], z[pos]
            angle1 = dir1[i, j, k, :]
            angle2 = dir2[i, j, k, :]
            if np.abs(angle1).sum() > 0:
                if np.abs(angle2).sum() > 0:
                    fractions = [frac1, 100-frac1]
                    angles = [angle1, angle2]
                    mevals = np.array(([0.002, 0.001, 0.001], [0.002, 0.001, 0.001]))
                # pdb.set_trace()
                else:
                    fractions = [100]
                    angles = [angle1]
                    mevals = np.array([[0.002, 0.001, 0.001]])
            else:
                if np.abs(angle2).sum() > 0:
                    fractions = [100]
                    angles = [angle2]
                    mevals = np.array([[0.002, 0.001, 0.001]])

            dwi[i, j, k, :], _ = multi_tensor(self.gtab,
                                              mevals=mevals,
                                              S0=S0,
                                              angles=angles,
                                              fractions=fractions,
                                              snr=snr)
        out = nib.Nifti1Image(dwi, self.affine, self.header)
        out.to_filename(fdwi)

    def gen_syn_multi_tensor_ffrac1(self, fdir1, fdir2, snr, fdwi,
                                    ffrac1):
        dir1 = nib.load(fdir1).get_data()
        dir2 = nib.load(fdir2).get_data()
        frac1img = nib.load(ffrac1).get_data()

        S0 = 1.
        dwi = np.zeros(list(self.mask.shape) + [len(self.bvals)])

        x, y, z = (self.mask > 0).nonzero()

        for pos in range(len(x)):
            i, j, k = x[pos], y[pos], z[pos]
            angle1 = dir1[i, j, k, :]
            angle2 = dir2[i, j, k, :]
            frac1 = frac1img[i, j, k]
            if np.abs(angle1).sum() > 0:
                if np.abs(angle2).sum() > 0:
                    fractions = [frac1, 100-frac1]
                    angles = [angle1, angle2]
                    mevals = np.array(([0.002, 0.001, 0.001], [0.002, 0.001, 0.001]))
                # pdb.set_trace()
                else:
                    fractions = [100]
                    angles = [angle1]
                    mevals = np.array([[0.002, 0.001, 0.001]])
            else:
                if np.abs(angle2).sum() > 0:
                    fractions = [100]
                    angles = [angle2]
                    mevals = np.array([[0.002, 0.001, 0.001]])

            dwi[i, j, k, :], _ = multi_tensor(self.gtab,
                                              mevals=mevals,
                                              S0=S0,
                                              angles=angles,
                                              fractions=fractions,
                                              snr=snr)
        out = nib.Nifti1Image(dwi, self.affine, self.header)
        out.to_filename(fdwi)

    def gen_syn_multi_tensor_flexF(self, fdir1, fdir2, snr, fdwi,
                                   ffrac1, frac1=70, frac1std=5):
        dir1 = nib.load(fdir1).get_data()
        dir2 = nib.load(fdir2).get_data()

        S0 = 1.
        dwi = np.zeros(list(self.mask.shape) + [len(self.bvals)])

        ##########
        # ind1 = np.zeros(list(self.mask.shape))
        # ind2 = np.zeros(list(self.mask.shape))
        ##########
        
        x, y, z = (self.mask > 0).nonzero()

        flex_f1 = np.random.normal(frac1, frac1std, self.mask.shape)
        flex_f1 = flex_f1 * self.mask

        for pos in range(len(x)):
            i, j, k = x[pos], y[pos], z[pos]
            angle1 = dir1[i, j, k, :]
            angle2 = dir2[i, j, k, :]
            if np.abs(angle1).sum() > 0:
                ########
                # ind1[i, j, k] = 1
                ########
                if np.abs(angle2).sum() > 0:
                    fractions = [flex_f1[i, j, k], 100-flex_f1[i, j, k]]
                    angles = [angle1, angle2]
                    mevals = np.array(([0.002, 0.001, 0.001], [0.002, 0.001, 0.001]))
                    #########
                    # ind2[i, j, k] = 1
                    #########
                # pdb.set_trace()
                else:
                    flex_f1[i, j, k] = 100
                    fractions = [100]
                    angles = [angle1]
                    mevals = np.array([[0.002, 0.001, 0.001]])
            else:
                if np.abs(angle2).sum() > 0:
                    ########
                    # ind1[i, j, k] = 1
                    ########
                    flex_f1[i, j, k] = 100
                    fractions = [100]
                    angles = [angle2]
                    mevals = np.array([[0.002, 0.001, 0.001]])

            dwi[i, j, k, :], _ = multi_tensor(self.gtab,
                                              mevals=mevals,
                                              S0=S0,
                                              angles=angles,
                                              fractions=fractions,
                                              snr=snr)
        out = nib.Nifti1Image(dwi, self.affine, self.header)
        out.to_filename(fdwi)
        out = nib.Nifti1Image(flex_f1, self.affine, self.header)
        out.to_filename(ffrac1)
        # out = nib.Nifti1Image(ind1, self.affine, self.header)
        # out.to_filename('./gndTruth_ind1.nii.gz')
        # out = nib.Nifti1Image(ind2, self.affine, self.header)
        # out.to_filename('./gndTruth_ind2.nii.gz')

    def dwi2sh(self, fdwi, sh_order, fsh):
        dwi = nib.load(fdwi).get_data()

        where_b0s = lazy_index(self.gtab.b0s_mask)
        where_dwi = lazy_index(~self.gtab.b0s_mask)
        b0_mean = np.mean(dwi[..., where_b0s], axis=-1)
        dwi_norm = np.divide(dwi[..., where_dwi], b0_mean[..., np.newaxis])

        x, y, z = self.gtab.gradients[where_dwi].T
        r, theta, phi = cart2sphere(x, y, z)
        B_dwi, m, n = real_sym_sh_mrtrix(sh_order,
                                         theta[:, None],
                                         phi[:, None])
        # pdb.set_trace()
        x, y, z = (self.mask > 0).nonzero()

        no_param = int((sh_order + 1) * (sh_order + 2) / 2)
        dwi_sh = np.zeros(list(self.mask.shape) + [no_param])
        for pos in range(len(x)):
            i, j, k = x[pos], y[pos], z[pos]
            dwi_sh[i, j, k, :] = np.linalg.lstsq(B_dwi,
                                                 dwi_norm[i, j, k, :],
                                                 rcond=None)[0]

        out = nib.Nifti1Image(dwi_sh, self.affine, self.header)
        out.to_filename(fsh)
        # out = nib.Nifti1Image(dwi_norm, self.affine, self.header)
        # out.to_filename('./dwi_norm.nii')

    def fodf2sh(self, fdir1, fdir2, ffrac1, sh_order, ffodfsh):
        dir1 = nib.load(fdir1).get_data()
        dir2 = nib.load(fdir2).get_data()
        frac1 = nib.load(ffrac1).get_data()

        frac1 = frac1 / 100

        x, y, z = (self.mask > 0).nonzero()
        no_param = int((sh_order + 1) * (sh_order + 2) / 2)
        fodf_sh = np.zeros(list(self.mask.shape) + [no_param])

        for pos in range(len(x)):
            i, j, k = x[pos], y[pos], z[pos]
            angle1 = dir1[i, j, k, :]
            angle2 = dir2[i, j, k, :]

            if np.abs(angle1).sum() > 0:
                if np.abs(angle2).sum() > 0:
                    data = np.array((frac1[i, j, k], 1-frac1[i, j, k]))
                    fiber = np.stack((dir1[i, j, k, :], dir2[i, j, k, :]), axis=0)
                    # pdb.set_trace()
                else:
                    data = np.array([frac1[i, j, k]])
                    fiber = np.array([dir1[i, j, k, :]])
            else:
                if np.abs(angle2).sum() > 0:
                    data = np.array([frac1[i, j, k]])
                    fiber = np.array([dir2[i, j, k, :]])

            sphere = Sphere(xyz=fiber)
            fodf_sh[i, j, k, :] = sf_to_sh(data, sphere, sh_order, "tournier07")
            # pdb.set_trace()
        out = nib.Nifti1Image(fodf_sh, self.affine, self.header)
        out.to_filename(ffodfsh)


def rotate_vector_img(fdir, fmask, fout, rot_axis='z', rot_degrees=90):
    from scipy.spatial.transform import Rotation as R

    dir_orig = nib.load(fdir)
    affine = dir_orig.affine
    header = dir_orig.header

    dir_orig = dir_orig.get_data()
    mask = nib.load(fmask).get_data()

    dir_rot = np.zeros(dir_orig.shape, dtype=dir_orig.dtype)
    rot = R.from_euler(rot_axis, rot_degrees, degrees=True)
    x, y, z = (mask > 0).nonzero()
    for pos in range(len(x)):
        i, j, k = x[pos], y[pos], z[pos]
        dir_rot[i, j, k, :] = rot.apply(dir_orig[i, j, k, :])
        # pdb.set_trace()

    out = nib.Nifti1Image(dir_rot, affine, header)
    out.to_filename(fout)


def reorient(input_image, target_orient='RAI'):
    orient_dict = {'R': 'L', 'A': 'P', 'I': 'S', 'L': 'R', 'P': 'A', 'S': 'I'}
    obj = nib.load(input_image)
    target_orient = [orient_dict[char] for char in target_orient]
    if nib.aff2axcodes(obj.affine) != tuple(target_orient):
        orig_ornt = nib.orientations.io_orientation(obj.affine)
        targ_ornt = nib.orientations.axcodes2ornt(target_orient)
        ornt_xfm = nib.orientations.ornt_transform(orig_ornt, targ_ornt)

        affine = obj.affine.dot(nib.orientations.inv_ornt_aff(ornt_xfm, obj.shape))
        data = nib.orientations.apply_orientation(obj.dataobj, ornt_xfm)
        obj_new = nib.Nifti1Image(data, affine, obj.header)
    else:
        obj_new = obj
    obj_new.to_filename(input_image.replace('.nii', '_reorient.nii'))
