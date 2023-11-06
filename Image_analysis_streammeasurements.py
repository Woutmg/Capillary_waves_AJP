# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:18:52 2021

@author: Wout M. Goesaert

This file contains the image analysis used for Goesaert & Logman (2023).
For questions about the code or suggestions, please contact the authors.
Wout Goesaert (first correspondent): wout.goesaert@gmail.com
Paul Logman (second correspondent: Logman@physics.leidenuniv.nl


There are a 10 variables that we gave seemingly arbitrary values,
"magic numbers". These are values that only work with our specific dataset of
images. They are specific to our lighting, image size, pixel scale, etc...
Therefore, when recreating our results with this analysis tool and
your own data, these values will probably need to be adapted by testing.
Testing can be done by setting (extensive/master)plot=True
List of "magic numbers" and their meaning:
clip_margin         - the photos are a silhouette of stream. By clipping at
                      this value we remove the noise in the dark background.

rod_condition       - a fixed threshold to define how strong the rod shows up
                      as a rise in brightness in the hor_profile_diff array

tube_condition      - a fixed threshold to define how strong the tube shows
                      up as a decline in brightness in the hor_profile_diff
                      array

streamcondition     - a fixed threshold to define the approximate brightness
                      level at which a contour would define the edge of
                      the stream

dx_ref1 and dx_ref2 - fixed distances from the top and bottom of the stream
                      at which the centre of the stream is calculated
                      to rotate the image

refhalfbandwidth    - defines the half-size of the vertical band around the
                      centre of the stream at which the stream edge could
                      be located

stream_searchwidth  - same as refhalfbandwidth but for edge_finder instead of
                      for the image rotator

edgeband_width      - a very small width around the rough edges of the stream
                      which is used to calculate a more precise location of
                      the edge (more precise than a single pixel)

moving_average_size - to reduce noise in the refined edge profile, smoothing
                      is done using a moving average, this needs to be set
                      significantly smaller than the wavelength!

switch_2ndwav_param - When the stream velocity becomes very high and the
                      wavelengths very small, the first wave becomes embedded
                      in the part of the stream where the water already
                      starts spreading out, becoming wider again due to the
                      water slowing down near the rod. To avoid measuring
                      in this regime, we switch to the next wavelength up
                      stream and call that wavelength the first wavelength
                      This parameter decides below what wavelength the switch
                      is made. When switching, the user is notified.
"""
# Importing needed packages:
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, os.path
from sys import exit
import scipy.signal as sc
import warnings

def point_rotator(vector, angle, middle):
    """This function rotates vectors around the middle point given a certain angle."""
    rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    vector_rotated = middle + rot_matrix.dot(vector-middle)
    return vector_rotated

def image_rotator(im, clipmargin=0.5, before_nozzle=5000, rod_condition=5e3, tube_condition=-3.55e3, streamcondition=250,
            dx_ref1=150, dx_ref2=-400, refhalfbandwidth=5, plot=False):
    """
    This function takes a stream image, finds two points in the centre
    of that stream and then rotates it such that the stream stands perfectly vertical.
    Set plot=True if you want to get diagnostic plots for testing.
    """
    im_bw = np.sum(np.array(im), axis=2) # Image is turned into greyscale numpy array
    im_clipped = np.clip(im_bw, 0., clipmargin * np.max(im_bw)) # Clipped version -> no noise in clipped dark parts
    im_shape = np.shape(im_bw)

    x_array = np.arange(0, im_shape[1])
    y_array = np.arange(0, im_shape[0])

    hor_profile = np.zeros(im_shape[1])
    refp1_profile = np.zeros(im_shape[0])
    refp2_profile = np.zeros(im_shape[0])

    # The image is summed vertically to find rough horizontal profile:
    for i in range(im_shape[1]):
        hor_profile[i] = np.sum(im_clipped[:, i])
    hor_profile_diff = np.diff(hor_profile)

    # This horizontal profile is used to find the rod and nozzle locations:
    rod_mask = np.where(hor_profile_diff[:before_nozzle] > rod_condition)
    rod_top = int((np.min(rod_mask) + np.max(rod_mask)) / 2)
    
    # The location of the nozzle is most recognizable from the location
    # of a handle connected to the nozzle which has a fixed distance from the
    # start of the stream
    tube_handle = rod_top + np.argmin(hor_profile_diff[rod_top:])

    # x coords are defined wrt the rod and nozzle as points which can be used to straighten the image:
    refpoint_1_x = rod_top + dx_ref1
    refpoint_2_x = tube_handle + dx_ref2

    # Vertical profile is obtained in region around x coords of repoints to find the edges of the stream:
    for i in range(im_shape[0]):
        refp1_profile[i] = np.sum(im_clipped[i, refpoint_1_x - refhalfbandwidth:refpoint_1_x + refhalfbandwidth])
        refp2_profile[i] = np.sum(im_clipped[i, refpoint_2_x - refhalfbandwidth:refpoint_2_x + refhalfbandwidth])
    refp1_profile_diff = np.diff(refp1_profile)
    refp2_profile_diff = np.diff(refp2_profile)

    # The y-coords (middle of the stream) are calculated for the given x-coords of the refpoints:
    stream_refp1_left_mask = np.where(refp1_profile_diff < -streamcondition)
    stream_refp1_left = np.min(stream_refp1_left_mask)
    stream_refp1_right_mask = np.where(refp1_profile_diff > streamcondition)
    stream_refp1_right = np.max(stream_refp1_right_mask)
    stream_refp1 = (stream_refp1_left + stream_refp1_right) / 2

    stream_refp2_left_mask = np.where(refp2_profile_diff < -streamcondition)
    stream_refp2_left = np.min(stream_refp2_left_mask)
    stream_refp2_right_mask = np.where(refp2_profile_diff > streamcondition)
    stream_refp2_right = np.max(stream_refp2_right_mask)
    stream_refp2 = (stream_refp2_left + stream_refp2_right) / 2

    refpoint_1_y = (stream_refp1_left + stream_refp1_right)/2
    refpoint_2_y = (stream_refp2_left + stream_refp2_right)/2

    # We now have two refpoints in the vertical center of the stream at the horizontal top and bottom:
    refpoint_1, refpoint_2 = (refpoint_1_x, refpoint_1_y), (refpoint_2_x, refpoint_2_y)

    # We can use these refpoints to straighten the image:
    angle_stream = np.degrees(np.arctan((stream_refp2 - stream_refp1) / (refpoint_2_x - refpoint_1_x)))
    im_rotated = im.rotate(angle_stream)

    if plot:
        plt.plot(x_array[1:], 100 * hor_profile_diff)
        plt.plot(x_array, hor_profile)
        plt.vlines([rod_top, tube_handle], 0., 1.7e6, color='blue')
        plt.vlines([refpoint_1_x, refpoint_2_x], 0., 1.7e6, color='red')
        plt.hlines(1e2*tube_condition, 0., 6000)
        plt.xlabel('x coord [pix]')
        plt.ylabel('profile (value, 100*value for diff)')
        plt.title('Selected refpoints on horizontal profile')
        plt.show()

        plt.figure(figsize=(9, 6))
        plt.imshow(im_bw, cmap='gray')
        plt.vlines([rod_top, tube_handle], 0, 4000, color='blue')
        plt.vlines([refpoint_1_x, refpoint_2_x], 0, 4000, color='red')
        plt.plot([refpoint_1_x, refpoint_2_x], [refpoint_1_y, refpoint_2_y], color='green')
        plt.xlabel('x coord [pix]')
        plt.ylabel('y coord [pix]')
        plt.title('Selected refpoints on image')
        plt.show()

        plt.plot(y_array, refp1_profile, color='red', label='Profile 1 (low)')
        plt.plot(y_array[1:], 10 * refp1_profile_diff, color='red', linestyle='dotted')
        plt.vlines([stream_refp1_left, stream_refp1, stream_refp1_right], -6000, 6000, color='red', linestyle='dashed')
        plt.plot(y_array, refp2_profile, color='blue', label='Profile 2 (high)')
        plt.plot(y_array[1:], 10 * refp2_profile_diff, color='blue', linestyle='dotted')
        plt.vlines([stream_refp2_left, stream_refp2, stream_refp2_right], -6000, 6000, color='blue', linestyle='dashed')
        plt.xlim(stream_refp2_left-100, stream_refp2_right+100)
        plt.xlabel('y coord [pix]')
        plt.ylabel('profile (value, 10*value for diff)')
        plt.title('Selected cross sections on image')
        plt.legend()
        plt.show()

        plt.figure(figsize=(9, 6))
        plt.imshow(im_rotated)
        plt.xlabel('x coord [pix]')
        plt.ylabel('y coord [pix]')
        plt.title('Rotated image')
        plt.show()

        print('The stream has an angle of ' + str(angle_stream) + ' degrees.')

    return (im_rotated, angle_stream, refpoint_1, refpoint_2, rod_top)

def edge_finder(image_rotated_bw, image_refpoint_1, image_refpoint_2, image_angle, middle, stream_searchwidth=300,\
                edgeband_width=7, moving_average_size=10, plot=False, extensiveplot=False):
    """
    This function takes an image of a stream, assumes it is straight,
    i.e. that the vector of gravity points down, and calculates the stream profile.
    Set plot=True or even extensiveplot=True if you want to get diagnostic plots.
    """

    # Again, clipping is performed to reduce noise. Trimmed means clipped to a lesser degree:
    image_clipped = np.clip(image_rotated_bw, 0.4 * np.max(image_rotated_bw), 0.5 * np.max(image_rotated_bw))
    image_trimmed = np.clip(image_rotated_bw, 0.2 * np.max(image_rotated_bw), 0.6 * np.max(image_rotated_bw))

    # The reference points are rotated to match the rotated, straight, image:
    image_refpoint_1_rotated = point_rotator(image_refpoint_1, np.deg2rad(image_angle), middle).astype(int)
    image_refpoint_2_rotated = point_rotator(image_refpoint_2, np.deg2rad(image_angle), middle).astype(int)

    if extensiveplot:
        # If wanted, steps can be plotted:
        plt.figure(figsize=(30, 20))
        plt.imshow(image_clipped)
        plt.vlines([image_refpoint_1_rotated[0], image_refpoint_2_rotated[0]], 0, 4000, color='red',
                   linewidth=5)
        plt.plot([image_refpoint_1_rotated[0], image_refpoint_2_rotated[0]],
                 [image_refpoint_1_rotated[1], image_refpoint_2_rotated[1]], color='red', linewidth=5)
        plt.title('Check that image is rotated correctly')
        plt.show()

    # Arrays are prepared:
    x_array_profile = np.arange(image_refpoint_1_rotated[0] - 150, image_refpoint_2_rotated[0] + 205)
    # We look 150 x-coords underneath refpoint 1 and 205 x-coords above refpoint 2
    profile_left = np.zeros(355 + image_refpoint_2_rotated[0] - image_refpoint_1_rotated[0])
    profile_right = np.zeros(355 + image_refpoint_2_rotated[0] - image_refpoint_1_rotated[0])
    profile_left_course = np.zeros(355 + image_refpoint_2_rotated[0] - image_refpoint_1_rotated[0])
    profile_right_course = np.zeros(355 + image_refpoint_2_rotated[0] - image_refpoint_1_rotated[0])

    # For every height, the profile is calculated across the stream:
    for i in range(len(profile_left)):
        # Derivatives are calculated to find the edge.
        diff_clipped = np.diff(image_clipped[:, image_refpoint_1_rotated[0] - 150 + i])
        diff_trimmed = np.diff(image_trimmed[:, image_refpoint_1_rotated[0] - 150 + i])

        if i == 200 and extensiveplot:
            plt.plot(image_clipped[:, image_refpoint_1_rotated[0] - 150 + i][
                     image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[1] + stream_searchwidth], label='clipped profile')
            plt.plot(image_trimmed[:, image_refpoint_1_rotated[0] - 150 + i][
                     image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[1] + stream_searchwidth], label='trimmed profile')
            plt.plot(10 * diff_clipped[image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[
                                                                                            1] + stream_searchwidth], label='clipped diff')
            plt.plot(10 * diff_trimmed[image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[
                                                                                            1] + stream_searchwidth], label='trimmed diff')
            plt.vlines([stream_searchwidth], min(diff_clipped), max(diff_clipped), color='red')
            plt.title('Horizontal stream profile at one height with derivatives')
            plt.show()

        # A rough search of the edge is made by just searching for the maximal point in the gradient:
        stream_left_coarse = image_refpoint_1_rotated[1] - stream_searchwidth + np.argmin(\
            diff_clipped[image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[1]])
        stream_right_coarse = image_refpoint_1_rotated[1] + np.argmax(diff_clipped[image_refpoint_1_rotated[1]: \
            image_refpoint_1_rotated[1] + stream_searchwidth])

        # In some rare cases, this can go wrong. We thus apply a trick:
        # If the point is too dissimilar to its neighbouring edge points,
        # it is considered wrong and we look for the next maximal gradient.
        if i != 0:
            difference_left = np.abs(stream_left_coarse - profile_left[i - 1])
            difference_right = np.abs(stream_right_coarse - profile_right[i - 1])

            if difference_left > 20:
                # If the point differs by more than 20 pixels, it is considered wrong.
                diff_clipped[stream_left_coarse] = 0
                stream_left_coarse = image_refpoint_1_rotated[1] - stream_searchwidth + np.argmin( \
                    diff_clipped[image_refpoint_1_rotated[1] - stream_searchwidth:image_refpoint_1_rotated[1]])

            if difference_right > 20:
                diff_clipped[stream_right_coarse] = 0
                stream_right_coarse = image_refpoint_1_rotated[1] + np.argmax(diff_clipped[image_refpoint_1_rotated[1]:\
                                                                    image_refpoint_1_rotated[1] + stream_searchwidth])

        # Next, to calculate the location of the edge of the stream more precisely,
        # a narrow band is defined around the edge in which we will perform our analysis:
        stream_left_edgeband = diff_trimmed[stream_left_coarse - edgeband_width:stream_left_coarse + edgeband_width]
        stream_right_edgeband = diff_trimmed[stream_right_coarse - edgeband_width:stream_right_coarse + edgeband_width]

        sum_left = np.sum(stream_left_edgeband)
        sum_right = np.sum(stream_right_edgeband)

        # The refined edge location is found by calculating the "center of mass"
        # with the "mass" here being the gradient amplitude. This assumes we have a gaussian blurr on the image:
        if sum_left == 0:
            # This deals with rare errors where the course edge was still not right
            stream_left = stream_left_coarse
        else:
            # Calculation of the "center of mass"
            stream_left = stream_left_coarse + np.sum(
                np.arange(-edgeband_width, edgeband_width) * stream_left_edgeband) / sum_left

        if sum_right == 0:
            # This deals with rare errors where the course edge was still not right
            stream_right = stream_right_coarse
        else:
            # Calculation of the "center of mass"
            stream_right = stream_right_coarse + np.sum(
                np.arange(-edgeband_width, edgeband_width) * stream_right_edgeband) / sum_right

        # Arrays are filled with found values
        profile_left[i], profile_right[i] = stream_left, stream_right
        profile_left_course[i], profile_right_course[i] = stream_left_coarse, stream_right_coarse

    # To further smooth the profiles, a moving average is taken.
    # This is done in order to later find the wavelength for the smallest of amplitudes.
    profile_left_smooth = np.convolve(profile_left, np.ones(moving_average_size) / moving_average_size, mode='same')
    profile_right_smooth = np.convolve(profile_right, np.ones(moving_average_size) / moving_average_size, mode='same')

    profile_middle = (profile_left + profile_right)/2

    if plot:
        plt.figure(figsize=(15, 10))
        plt.plot(profile_left_smooth, color='red', linewidth=3)
        plt.plot(profile_right_smooth, color='red', linewidth=3)
        plt.plot(profile_left_course, color='blue')
        plt.plot(profile_right_course, color='blue')
        plt.title('Check comparison rough to smooth edge')
        plt.show()

    if extensiveplot:
        plt.figure(figsize=(15, 10), dpi=400)
        plt.imshow(image_rotated_bw, cmap='gray')
        plt.plot(x_array_profile, profile_left_smooth, color='red', linewidth=0.5)
        plt.plot(x_array_profile, profile_middle, color='orange', linewidth=0.5)
        plt.plot(x_array_profile, profile_right_smooth, color='blue', linewidth=0.5)
        plt.vlines([image_refpoint_1_rotated[0] - 150 + 123], 0, 4000)
        plt.scatter([image_refpoint_1_rotated[0], image_refpoint_2_rotated[0]],
                [image_refpoint_1_rotated[1], image_refpoint_2_rotated[1]], color='red',
                marker='x')
        plt.xlabel('x coord [pix]')
        plt.ylabel('y coord [pix]')
        plt.title('Check everything plotted on figure')
        plt.show()

    return(x_array_profile, profile_left, profile_right, profile_middle, profile_left_smooth, profile_right_smooth)

def image_analyser(map_location, scale_pix, grav, density, surftens, masterplot=True, switch_2ndwav_param=50):
    """
    This function takes a single height step and calculates the stream radius
    and wavelengths for the images available for that height.
    Set masterplot=True to get the most important diagnostic plot.
    """
    valid_images = [".jpg",".JPG"]

    image_list = []

    for f in os.listdir(map_location):
        ext = os.path.splitext(f)[1]

        if ext.lower() in valid_images:
            image = Image.open(os.path.join(map_location,f))
            image_list.append(image)

    # Empty lists are initiated:
    image_list_rotated_bw, image_list_angle = [], []
    list_refpoint_1, list_refpoint_2 = [], []
    list_rodtop = []

    wavelengths_measured_left_1st, wavelengths_measured_left_2nd, wavelengths_measured_left_3rd = [], [], []
    wavelengths_measured_right_1st, wavelengths_measured_right_2nd, wavelengths_measured_right_3rd = [], [], []

    radii_mean_left_1st, radii_mean_left_2nd, radii_mean_left_3rd = [], [], []
    radii_mean_right_1st, radii_mean_right_2nd, radii_mean_right_3rd = [], [], []
    radii_mean_left_1st_minmax, radii_mean_left_2nd_minmax, radii_mean_left_3rd_minmax = [], [], []
    radii_mean_right_1st_minmax, radii_mean_right_2nd_minmax, radii_mean_right_3rd_minmax = [], [], []

    # Default, we don't switch to second wave untill threshold is reached
    switchto2ndwav = False

    for n in range(len(image_list)):
        # For every image we rotate it, find the edge of the stream,
        # find the throughs and save three (if available) wavelengths
        # at each side of the stream:
        middle = np.array([np.shape(image_list[n])[1] / 2, np.shape(image_list[n])[0] / 2])

        image_rotated, image_angle, image_refpoint_1, image_refpoint_2, image_rodtop \
            = image_rotator(image_list[n], plot=True)

        image_rotated_bw = np.sum(np.array(image_rotated), axis=2)
        image_list_rotated_bw.append(image_rotated_bw)
        image_list_angle.append(image_angle)
        list_refpoint_1.append(image_refpoint_1)
        list_refpoint_2.append(image_refpoint_2)
        list_rodtop.append(image_rodtop)

        streamlen_approx = (355 + image_refpoint_2[0] - image_refpoint_1[0])/scale_pix
        
        # To find throughs(=peaks) in the stream profile, we need to do a
        # throughs searchvthis is done by comparing each part of the stream
        # to the neighbouring stream radii. The number of pixels we compare
        # with at either side is the peaksearch_order and this must be
        # significantly smaller than one wavelength. We calculate it here:
        wavelen_approx = np.pi*surftens/(density*grav*streamlen_approx)
        wavelen_pix_approx = wavelen_approx*scale_pix
        peaksearch_order = min(20, int(wavelen_pix_approx/6))
        if n==0:
            print('peaksearch order: ' + str(peaksearch_order))
            
        # The edge profiles are calculated:
        image_array_profile, image_profile_left, image_profile_right, image_profile_middle, image_profile_left_smooth\
        , image_profile_right_smooth = edge_finder(image_rotated_bw, image_refpoint_1, image_refpoint_2\
                                                   , image_angle, middle, moving_average_size=peaksearch_order, plot=True, extensiveplot=True)

        # The throughs are calculated:
        troughs_left = np.array(sc.argrelmax(image_profile_left_smooth, order=peaksearch_order))[0]
        troughs_right = np.array(sc.argrelmax(-image_profile_right_smooth, order=peaksearch_order))[0]

        # And from that the wavelengths
        wavelengths_pix_left = np.diff(troughs_left)
        wavelengths_pix_right = np.diff(troughs_right)

        if masterplot and n==0:
            plt.imshow(image_rotated_bw, cmap='gray')
            plt.scatter(image_array_profile[troughs_left], image_profile_left_smooth[troughs_left]-5, s=30, color='blue')
            plt.scatter(image_array_profile[troughs_right], image_profile_right_smooth[troughs_right]-5, s=30, color='blue')
            plt.plot(image_array_profile, image_profile_left_smooth-5, color='red')
            plt.plot(image_array_profile, image_profile_right_smooth-5, color='orange')
            plt.xlim(np.min(image_array_profile),(np.min(image_array_profile)+np.max(image_array_profile))/2)
            plt.ylim(1500,2800)
            plt.show()

        image_wavelength_measured_left_1st = wavelengths_pix_left[0] / scale_pix
        wave_profile_left_1st = image_profile_left[troughs_left[0]:troughs_left[1]] \
                                - image_profile_middle[troughs_left[0]:troughs_left[1]]
        image_wavelength_measured_right_1st = wavelengths_pix_right[0] / scale_pix
        wave_profile_right_1st = image_profile_right[troughs_right[0]:troughs_right[1]] \
                                 - image_profile_middle[troughs_right[0]:troughs_right[1]]

        # At each wave, we now calculate the mean radius of the stream over that wave:
        wave_meanradius_left_1st = np.nanmean(wave_profile_left_1st)
        wave_meanradius_right_1st = np.nanmean(wave_profile_right_1st)
        wave_meanradius_left_1st_minmax = (np.nanmin(wave_profile_left_1st)+np.nanmax(wave_profile_left_1st))/2
        wave_meanradius_right_1st_minmax = (np.nanmin(wave_profile_right_1st)+np.nanmax(wave_profile_right_1st))/2


        try:
            # This try statement is needed because sometimes no second/third wavelength exists
            image_wavelength_measured_left_2nd = wavelengths_pix_left[1]/scale_pix
            wave_profile_left_2nd = image_profile_left[troughs_left[1]:troughs_left[2]] \
                                    - image_profile_middle[troughs_left[1]:troughs_left[2]]
            image_wavelength_measured_right_2nd = wavelengths_pix_right[1]/scale_pix
            wave_profile_right_2nd = image_profile_right[troughs_right[1]:troughs_right[2]] \
                                     - image_profile_middle[troughs_right[1]:troughs_right[2]]

            wave_meanradius_left_2nd = np.nanmean(wave_profile_left_2nd)
            wave_meanradius_right_2nd = np.nanmean(wave_profile_right_2nd)
            wave_meanradius_left_2nd_minmax = (np.nanmin(wave_profile_left_2nd)+np.nanmax(wave_profile_left_2nd))/2
            wave_meanradius_right_2nd_minmax = (np.nanmin(wave_profile_right_2nd)+np.nanmax(wave_profile_right_2nd))/2


        except:
            image_wavelength_measured_left_2nd = np.nan
            wave_profile_left_2nd = np.array([])
            image_wavelength_measured_right_2nd = np.nan
            wave_profile_right_2nd = np.array([])

            wave_meanradius_left_2nd = np.nan
            wave_meanradius_right_2nd = np.nan
            wave_meanradius_left_2nd_minmax = np.nan
            wave_meanradius_right_2nd_minmax = np.nan

        try:
            image_wavelength_measured_left_3rd = wavelengths_pix_left[2]/scale_pix
            wave_profile_left_3rd = image_profile_left[troughs_left[2]:troughs_left[3]] \
                                    - image_profile_middle[troughs_left[2]:troughs_left[3]]
            image_wavelength_measured_right_3rd = wavelengths_pix_right[2]/scale_pix
            wave_profile_right_3rd = image_profile_right[troughs_right[2]:troughs_right[3]] \
                                     - image_profile_middle[troughs_right[2]:troughs_right[3]]

            wave_meanradius_left_3rd = np.nanmean(wave_profile_left_3rd)
            wave_meanradius_right_3rd = np.nanmean(wave_profile_right_3rd)
            wave_meanradius_left_3rd_minmax = (np.nanmin(wave_profile_left_3rd)+np.nanmax(wave_profile_left_3rd))/2
            wave_meanradius_right_3rd_minmax = (np.nanmin(wave_profile_right_3rd)+np.nanmax(wave_profile_right_3rd))/2

        except:
            image_wavelength_measured_left_3rd = np.nan
            wave_profile_left_3rd = np.array([])
            image_wavelength_measured_right_3rd = np.nan
            wave_profile_right_3rd = np.array([])

            wave_meanradius_left_3rd = np.nan
            wave_meanradius_right_3rd = np.nan
            wave_meanradius_left_3rd_minmax = np.nan
            wave_meanradius_right_3rd_minmax = np.nan

        wavelengths_measured_left_1st.append(image_wavelength_measured_left_1st)
        wavelengths_measured_left_2nd.append(image_wavelength_measured_left_2nd)
        wavelengths_measured_left_3rd.append(image_wavelength_measured_left_3rd)

        wavelengths_measured_right_1st.append(image_wavelength_measured_right_1st)
        wavelengths_measured_right_2nd.append(image_wavelength_measured_right_2nd)
        wavelengths_measured_right_3rd.append(image_wavelength_measured_right_3rd)

        radii_mean_left_1st.append(np.abs(wave_meanradius_left_1st)/scale_pix)
        radii_mean_left_2nd.append(np.abs(wave_meanradius_left_2nd)/scale_pix)
        radii_mean_left_3rd.append(np.abs(wave_meanradius_left_3rd)/scale_pix)

        radii_mean_left_1st_minmax.append(np.abs(wave_meanradius_left_1st_minmax)/scale_pix)
        radii_mean_left_2nd_minmax.append(np.abs(wave_meanradius_left_2nd_minmax)/scale_pix)
        radii_mean_left_3rd_minmax.append(np.abs(wave_meanradius_left_3rd_minmax)/scale_pix)

        radii_mean_right_1st.append(np.abs(wave_meanradius_right_1st)/scale_pix)
        radii_mean_right_2nd.append(np.abs(wave_meanradius_right_2nd)/scale_pix)
        radii_mean_right_3rd.append(np.abs(wave_meanradius_right_3rd)/scale_pix)

        radii_mean_right_1st_minmax.append(np.abs(wave_meanradius_right_1st_minmax)/scale_pix)
        radii_mean_right_2nd_minmax.append(np.abs(wave_meanradius_right_2nd_minmax)/scale_pix)
        radii_mean_right_3rd_minmax.append(np.abs(wave_meanradius_right_3rd_minmax)/scale_pix)

        if min(troughs_left[0], troughs_right[0]) < switch_2ndwav_param:
            # See info about this switch at begin of document behind "switch_2ndwav_param"
            switchto2ndwav = True
            print('Switching to second wavelength')

        if masterplot:
            print('left wavelength = ' + str(image_wavelength_measured_left_1st) \
                  + " " + str(image_wavelength_measured_left_2nd))
            print('right wavelength = ' + str(image_wavelength_measured_right_1st) \
                  + " " + str(image_wavelength_measured_right_2nd))

    if switchto2ndwav:
        print('Switch has happened.')

    if switchto2ndwav==False:
        radii_left = radii_mean_left_1st
        radii_right = radii_mean_right_1st
        wavelengths_left = wavelengths_measured_left_1st
        wavelengths_right = wavelengths_measured_right_1st

    else:
        try:
            radii_left = radii_mean_left_2nd
            radii_right = radii_mean_right_2nd
            wavelengths_left = wavelengths_measured_left_2nd
            wavelengths_right = wavelengths_measured_right_2nd
        except TypeError:
            print("Something went wrong. It might be the case that switch was made to 2nd wave "\
                  "but some image only had 1 wave available.")
            exit()

    radius_left = np.median(radii_left)
    radius_left_error = np.std(radii_left)/np.sqrt(len(radii_left))
    radius_right = np.median(radii_right)
    radius_right_error = np.std(radii_right)/np.sqrt(len(radii_right))
    radii_master = np.append(radii_left, radii_right)
    radius = np.median(radii_master)

    radius_error = np.std(radii_master)/np.sqrt(len(radii_master))

    radius_collection = np.array([radius_left, radius_left_error, radius_right, radius_right_error, radius, radius_error])
    radius_extensive = np.array([radii_mean_left_1st, radii_mean_left_2nd\
                                                , radii_mean_left_3rd, radii_mean_right_1st\
                                                , radii_mean_right_2nd, radii_mean_right_3rd\
                                                , radii_mean_left_1st_minmax, radii_mean_left_2nd_minmax\
                                                , radii_mean_left_3rd_minmax, radii_mean_right_1st_minmax\
                                                , radii_mean_right_2nd_minmax, radii_mean_right_3rd_minmax])

    wavelen_left = np.median(wavelengths_left)
    wavelen_left_error = np.std(wavelengths_left)/np.sqrt(len(wavelengths_left))
    wavelen_right = np.median(wavelengths_right)
    wavelen_right_error = np.std(wavelengths_right)/np.sqrt(len(wavelengths_right))
    wavelengths_master = np.append(wavelengths_left, wavelengths_right)
    wavelen = np.median(wavelengths_master)
    wavelen_error = np.std(wavelengths_master)/np.sqrt(len(wavelengths_master))

    wavelen_collection = np.array([wavelen_left, wavelen_left_error, wavelen_right, wavelen_right_error, wavelen, wavelen_error])
    wavelen_extensive = np.array([wavelengths_measured_left_1st, wavelengths_measured_left_2nd\
                                                , wavelengths_measured_left_3rd, wavelengths_measured_right_1st\
                                                , wavelengths_measured_right_2nd, wavelengths_measured_right_3rd])

    return(radius_collection, wavelen_collection, switchto2ndwav, radius_extensive, wavelen_extensive)

# Constants are defined:
g = 9.81  # 9.812
g_error = 0.01  # 0.001
water_d = 997
water_d_error = 1
water_surft = 72.8e-3
water_surft_error = 0.1e-3
diameter_nozzle = 6e-3
diameter_nozzle_error = 5e-5

# Stream flow rate is calculated from measurements:
volume_filled = 4e-4
volume_filled_error = 1e-6
time_filled_list = np.loadtxt("Flowrate_Measurement_Newdata.txt", delimiter=",", dtype=str).astype(float)

time_filled_N = len(time_filled_list) * len(time_filled_list[0])
time_filled = np.mean(time_filled_list)
time_filled_error = np.std(time_filled_list) / np.sqrt(time_filled_N)

flow_rate = volume_filled / (20 * time_filled)
flow_rate_error = flow_rate * (
            (((volume_filled_error / volume_filled) ** 2) + ((time_filled_error / 20 * time_filled) ** 2)) ** 0.5)

# The physical scale of each pixel is calculated:
cal_pixnum_list = np.array([4550, 4547, 4546, 4535])
cal_pixnum = np.mean(cal_pixnum_list)
cal_pixnum_error = np.std(cal_pixnum_list) / np.sqrt(len(cal_pixnum_list))
cal_length = 69.72 * 1e-3 #meter
cal_length_error = 0.01 * 1e-3
scale = cal_pixnum / cal_length
scale_error = scale * ((((cal_pixnum_error / cal_pixnum) ** 2) + ((cal_length_error / cal_length) ** 2)) ** 0.5)

# Wavelengths and average radii are extracted from images
heights = 33
stream_radii = np.zeros((heights, 9))
stream_radii_error = np.zeros((heights, 9))
stream_radii_1, stream_radii_1_minmax = np.zeros((heights, 9)), np.zeros((heights, 9))
stream_radii_2, stream_radii_2_minmax = np.zeros((heights, 9)), np.zeros((heights, 9))
stream_radii_3, stream_radii_3_minmax = np.zeros((heights, 9)), np.zeros((heights, 9))

stream_wavelen = np.zeros(heights)
stream_wavelen_error = np.zeros(heights)
stream_wavelen_1_left = np.zeros((heights, 9))
stream_wavelen_2_left = np.zeros((heights, 9))
stream_wavelen_3_left = np.zeros((heights, 9))
stream_wavelen_1_right = np.zeros((heights, 9))
stream_wavelen_2_right = np.zeros((heights, 9))
stream_wavelen_3_right = np.zeros((heights, 9))

switches = np.zeros(heights)

for heightnum in range(heights):
    heightnum = 13
    # We call the image analyser for each height step:
    print('Heightnum: ' + str(heightnum))
    path = r'C:\Users\woutg\OneDrive - Universiteit Leiden\Universiteit Leiden\Bachelor 1\EN\Paper\New Data\Wavelength Measurements\Height ' + str(heightnum + 1)

    radius_collection_test, wavelen_collection_test, wave_switch, radius_extensive_test\
        , wavelen_extensive_test = image_analyser(path, scale, g, water_d, water_surft\
        , masterplot=True, switch_2ndwav_param=45)

    stream_radii[heightnum], stream_radii_error[heightnum] = radius_collection_test[4], radius_collection_test[5]
    stream_wavelen[heightnum], stream_wavelen_error[heightnum] = wavelen_collection_test[4], wavelen_collection_test[5]
    switches[heightnum] = wave_switch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # At each wave, we calculate the mean radius of the stream over that wave:
        stream_radii_1[heightnum] = np.nanmean(np.array([radius_extensive_test[0],radius_extensive_test[3]]), axis=0)
        stream_radii_2[heightnum] = np.nanmean(np.array([radius_extensive_test[1],radius_extensive_test[4]]), axis=0)
        stream_radii_3[heightnum] = np.nanmean(np.array([radius_extensive_test[2],radius_extensive_test[5]]), axis=0)
        # Another method (easier for students to do by hand) is to approximate
        # the stream radius from the max and min radius along a wave.
        # This assumes the wave is sinusoidal. To test the introduced error,
        # we also calculate and save this:
        stream_radii_1_minmax[heightnum] = np.nanmean(np.array([radius_extensive_test[6],radius_extensive_test[9]]), axis=0)
        stream_radii_2_minmax[heightnum] = np.nanmean(np.array([radius_extensive_test[7],radius_extensive_test[10]]), axis=0)
        stream_radii_3_minmax[heightnum] = np.nanmean(np.array([radius_extensive_test[8],radius_extensive_test[11]]), axis=0)

    stream_wavelen_1_left[heightnum] = wavelen_extensive_test[0]
    stream_wavelen_2_left[heightnum] = wavelen_extensive_test[1]
    stream_wavelen_3_left[heightnum] = wavelen_extensive_test[2]
    stream_wavelen_1_right[heightnum] = wavelen_extensive_test[3]
    stream_wavelen_2_right[heightnum] = wavelen_extensive_test[4]
    stream_wavelen_3_right[heightnum] = wavelen_extensive_test[5]

stream_velocity = flow_rate/(np.pi*(stream_radii**2))
stream_velocity_1 = flow_rate/(np.pi*(stream_radii_1**2))
stream_velocity_2 = flow_rate/(np.pi*(stream_radii_2**2))
stream_velocity_3 = flow_rate/(np.pi*(stream_radii_3**2))

stream_velocity_error = np.sqrt((flow_rate_error/(np.pi*stream_radii**2))**2 \
                                + (2*flow_rate*stream_radii_error/(np.pi*stream_radii**3))**2)

# We save all wavelengths, velocities and radii
np.save('stream_vels.npy', stream_velocity)
np.save('stream_vels_1.npy', stream_velocity_1)
np.save('stream_vels_2.npy', stream_velocity_2)
np.save('stream_vels_3.npy', stream_velocity_3)

np.save('stream_vels_errs.npy', stream_velocity_error)
np.save('wavs.npy', stream_wavelen)
np.save('wavs_1_left.npy', stream_wavelen_1_left)
np.save('wavs_2_left.npy', stream_wavelen_2_left)
np.save('wavs_3_left.npy', stream_wavelen_3_left)
np.save('wavs_1_right.npy', stream_wavelen_1_right)
np.save('wavs_2_right.npy', stream_wavelen_2_right)
np.save('wavs_3_right.npy', stream_wavelen_3_right)

np.save('wavs_errs.npy', stream_wavelen_error)
np.save('radii.npy', stream_radii)
np.save('radii_1.npy', stream_radii_1)
np.save('radii_2.npy', stream_radii_2)
np.save('radii_3.npy', stream_radii_3)
np.save('radii_1_minmax.npy', stream_radii_1_minmax)
np.save('radii_2_minmax.npy', stream_radii_2_minmax)
np.save('radii_3_minmax.npy', stream_radii_3_minmax)

np.save('radii_errs.npy', stream_radii_error)

# The switches variable keeps track of at what heights a switch to the second
# wavelength is made (ignoring the first wavelength):
print('Switches:' + str(switches))