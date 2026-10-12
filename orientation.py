"""
orientation.py -- Coordinate conventions, rotations, and projection.

CONVENTIONS (defined here once, used everywhere) --------------------------

Origin: the central star.

Sky frame (right-handed) axes:
    x = EAST   (points left on a north-up/east-left image)
    y = NORTH  (points up on the image)
    z = LINE OF SIGHT, positive pointing AWAY from the observer.
        => a parcel with positive z-velocity is RECEDING  -> REDSHIFT  (+v)
           a parcel with negative z-velocity is APPROACHING -> BLUESHIFT (-v)

Because (x=east, y=north, z=away) and east is "left", the on-sky axes form a
right-handed set when viewed from BEHIND the sky plane (the usual astronomer's
convention). When we display an image we put north up and flip x so east is
to the left.

Body frame: each component is built with its SYMMETRY AXIS along body +y
(so a bipolar nebula's lobes run along body +/- y, "north-like" before tilting).

Inclination i: angle between the symmetry (lobe) axis and the plane of the sky.
    i = 0   -> axis lies in the sky plane (no radial-velocity separation of lobes)
    i = 90  -> axis points along the line of sight (lobes overlap on the sky,
               maximally separated in velocity)

Position angle PA: measured from NORTH through EAST (astronomical convention).

Body -> sky rotation:
    R(i, PA) = Rz(-PA) @ Rx(i)
where Rx is a rotation about the sky x-axis (east) and Rz about the sky z-axis
(line of sight). Applied to the body symmetry axis (0, 1, 0) this gives the
sky-frame axis direction
    a_hat = ( cos i * sin PA,   cos i * cos PA,   sin i ).
Check: i=0, PA=0 -> (0,1,0) = north (in sky plane). Increasing i tilts it toward
+z (receding). Increasing PA swings it from north toward east. (The sign choices
are verified numerically by Phase-2 validation tests 3 and 4.)

To evaluate a body-frame analytic model on a sky-frame grid we use the INVERSE
rotation R^T (rotations are orthogonal, so inverse = transpose):
    r_body = R^T @ r_sky.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# --- Elementary rotation matrices (right-handed, rotate column vectors) ----

def Rx(angle_rad):
    """Rotation about the sky x-axis (east) by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def Ry(angle_rad):
    """Rotation about the sky y-axis (north) by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def Rz(angle_rad):
    """Rotation about the sky z-axis (line of sight) by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def body_to_sky_matrix(inclination_deg, position_angle_deg):
    """Return the 3x3 rotation R mapping BODY coordinates to SKY coordinates.

    R(i, PA) = Rz(-PA) @ Rx(i).  See module docstring for the derivation.
    """
    i = np.deg2rad(inclination_deg)
    pa = np.deg2rad(position_angle_deg)
    return Rz(-pa) @ Rx(i)


def symmetry_axis_sky(inclination_deg, position_angle_deg):
    """Unit vector of the nebular symmetry axis in the sky frame.

    Equals R @ (0,1,0) = (cos i sin PA, cos i cos PA, sin i).
    """
    R = body_to_sky_matrix(inclination_deg, position_angle_deg)
    return R @ np.array([0.0, 1.0, 0.0])


def sky_to_body(points_sky, inclination_deg, position_angle_deg):
    """Map sky-frame points (..., 3) to body-frame points using R^T.

    points_sky may be an array of shape (N, 3) or (3,).
    """
    R = body_to_sky_matrix(inclination_deg, position_angle_deg)
    pts = np.asarray(points_sky, dtype=float)
    return pts @ R  # equivalent to (R.T @ pts.T).T since R is orthogonal


def body_to_sky(points_body, inclination_deg, position_angle_deg):
    """Map body-frame points (..., 3) to sky-frame points using R."""
    R = body_to_sky_matrix(inclination_deg, position_angle_deg)
    pts = np.asarray(points_body, dtype=float)
    return pts @ R.T


# --- Conventions diagram ---------------------------------------------------

def draw_conventions_diagram(save_path):
    """Draw and save a labelled diagram of the coordinate conventions."""
    fig, (ax_sky, ax_los) = plt.subplots(1, 2, figsize=(12, 5.5))

    # ----- Left panel: the sky plane as the observer sees it -----
    ax_sky.set_title("Sky plane (observer's view)\nnorth up, east left")
    ax_sky.set_xlim(-1.2, 1.2)
    ax_sky.set_ylim(-1.2, 1.2)
    ax_sky.set_aspect("equal")
    # North arrow (up, +y)
    ax_sky.annotate("", xy=(0, 1.0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax_sky.text(0.03, 1.02, "NORTH (+y)", va="bottom")
    # East arrow (left, +x) -- east is to the LEFT
    ax_sky.annotate("", xy=(-1.0, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax_sky.text(-1.02, 0.05, "EAST (+x)", ha="left")
    # Position angle arc from north toward east
    pa_deg = 160.0
    th = np.linspace(90, 90 + pa_deg, 100)  # start at north (90 deg in math frame), go toward east
    # In our display east is -x_display, north +y_display; we draw PA sweeping N->E
    r_arc = 0.55
    xs = -r_arc * np.sin(np.deg2rad(th - 90))  # east is left => negative display-x
    ys = r_arc * np.cos(np.deg2rad(th - 90))
    ax_sky.plot(xs, ys, color="tab:blue")
    ax_sky.text(-0.15, -0.35, f"PA = {pa_deg:.0f}°\n(N through E)",
                color="tab:blue", fontsize=9)
    # Projected lobe axis at PA=160
    a = np.deg2rad(pa_deg)
    # east(left)=-sin(PA) in display-x, north(up)=cos(PA) in display-y
    ax_sky.annotate("", xy=(-np.sin(a), np.cos(a)), xytext=(np.sin(a), -np.cos(a)),
                    arrowprops=dict(arrowstyle="-", lw=2.5, color="tab:red", alpha=0.7))
    ax_sky.text(-np.sin(a) * 1.05, np.cos(a) * 1.05, "projected\nlobe axis",
                color="tab:red", fontsize=9, ha="center")
    ax_sky.plot(0, 0, "*", color="gold", markersize=18, markeredgecolor="k")
    ax_sky.text(0.05, -0.12, "central star", fontsize=9)
    ax_sky.set_xlabel("display x  (east is LEFT)")
    ax_sky.set_ylabel("display y  (north up)")

    # ----- Right panel: line-of-sight / velocity convention -----
    ax_los.set_title("Line of sight & velocity sign\n(side view)")
    ax_los.set_xlim(-1.3, 1.3)
    ax_los.set_ylim(-1.2, 1.2)
    ax_los.set_aspect("equal")
    ax_los.plot(0, 0, "*", color="gold", markersize=18, markeredgecolor="k")
    # Observer on the left (negative z); +z points away (to the right)
    ax_los.annotate("", xy=(1.1, 0), xytext=(-1.1, 0),
                    arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax_los.text(1.12, 0.03, "+z (line of sight,\naway from observer)", fontsize=8, va="bottom")
    ax_los.text(-1.1, -0.18, "observer", fontsize=9)
    ax_los.annotate("", xy=(-0.8, 0.0), xytext=(-0.5, 0.0),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
    # Inclination: tilt axis by i from sky plane (vertical dashed = sky plane)
    ax_los.plot([0, 0], [-1.0, 1.0], "k--", lw=1, alpha=0.5)
    ax_los.text(0.02, 1.0, "sky plane", fontsize=8, alpha=0.7)
    inc = np.deg2rad(60.0)
    # axis tilted by i from the vertical sky plane toward +z
    ax_los.annotate("", xy=(np.sin(inc), np.cos(inc)), xytext=(-np.sin(inc), -np.cos(inc)),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color="tab:red"))
    ax_los.text(np.sin(inc) * 1.05, np.cos(inc) * 1.05, "lobe axis", color="tab:red",
                fontsize=9)
    # inclination arc
    th = np.linspace(90, 90 - 60, 60)
    ax_los.plot(0.45 * np.cos(np.deg2rad(th)), 0.45 * np.sin(np.deg2rad(th)),
                color="tab:green")
    ax_los.text(0.32, 0.55, "i = 60°", color="tab:green", fontsize=9)
    # Redshift / blueshift labels
    ax_los.text(0.62, 0.78, "receding\n(+v, redshift)", color="red", fontsize=8)
    ax_los.text(-1.05, -0.78, "approaching\n(-v, blueshift)", color="blue", fontsize=8)
    ax_los.set_xlabel("z (line of sight)")
    ax_los.set_ylabel("sky-plane direction")

    fig.suptitle("NGC 6905 model coordinate conventions", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path
