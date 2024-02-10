import torch


def theta_phi_to_xyz(theta, phi):
    xyz = torch.cat((
        torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)
    ), dim=1)
    return xyz

def roll_T(n, xyz, gamma=0):
    gamma = gamma / 180 * torch.pi
    n11 = (n[...,0] ** 2) * (1 - torch.cos(gamma)) + torch.cos(gamma)
    n12 = n[...,0] * n[...,1] * (1 - torch.cos(gamma)) - n[...,2] * torch.sin(gamma)
    n13 = n[...,0] * n[...,2] * (1 - torch.cos(gamma)) + n[...,1] * torch.sin(gamma)

    n21 = n[...,0] * n[...,1] * (1 - torch.cos(gamma)) + n[...,2] * torch.sin(gamma)
    n22 = (n[...,1] ** 2) * (1 - torch.cos(gamma)) + torch.cos(gamma)
    n23 = n[...,1] * n[...,2] * (1 - torch.cos(gamma)) - n[...,0] * torch.sin(gamma)

    n31 = n[...,0] * n[...,2] * (1 - torch.cos(gamma)) - n[...,1] * torch.sin(gamma)
    n32 = n[...,1] * n[...,2] * (1 - torch.cos(gamma)) + n[...,0] * torch.sin(gamma)
    n33 = (n[...,2] ** 2) * (1 - torch.cos(gamma)) + torch.cos(gamma)

    x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]

    xx = torch.diagonal(n11 * x + n12 * y + n13 * z).unsqueeze(1)
    yy = torch.diagonal(n21 * x + n22 * y + n23 * z).unsqueeze(1)
    zz = torch.diagonal(n31 * x + n32 * y + n33 * z).unsqueeze(1)

    return torch.cat((xx, yy, zz), dim=1)

def roArrayVector(theta, phi, v, ang):
    c_xyz = theta_phi_to_xyz(theta, phi)
    p_xyz = v
    pp_xyz = roll_T(c_xyz, p_xyz, ang)
    return pp_xyz

class Sph:
    '''Unbiased IoU Computation for Spherical Rectangles'''

    def __init__(self):
        self.visited, self.trace, self.pot = [], [], []

    def area(self, fov_x, fov_y):
        '''Area Computation'''
        return 4 * torch.arccos(-torch.sin(fov_x / 2) * torch.sin(fov_y / 2)) - 2 * torch.pi
    
    def getNormal(self, bbox):
        device = bbox.device  # Ensure all tensors are on the same device as bbox

        bbox[:, [4]] = 0

        theta, phi, fov_x_half, fov_y_half, angle = bbox[:, [
            0]], bbox[:, [1]], bbox[:, [2]] / 2, bbox[:, [3]] / 2, bbox[:, [4]]

        V_lookat = torch.cat((
            torch.sin(phi) * torch.cos(theta), torch.sin(phi) *
            torch.sin(theta), torch.cos(phi)
        ), dim=1)
        V_right = torch.cat(
            (-torch.sin(theta), torch.cos(theta), torch.zeros(theta.shape, device=device)), dim=1)
        V_up = torch.cat((
            -torch.cos(phi) * torch.cos(theta), -torch.cos(phi) *
            torch.sin(theta), torch.sin(phi)
        ), dim=1)

        N_left = -torch.cos(fov_x_half) * V_right + torch.sin(fov_x_half) * V_lookat
        N_right = torch.cos(fov_x_half) * V_right + torch.sin(fov_x_half) * V_lookat
        N_up = -torch.cos(fov_y_half) * V_up + torch.sin(fov_y_half) * V_lookat
        N_down = torch.cos(fov_y_half) * V_up + torch.sin(fov_y_half) * V_lookat

        N_left = roArrayVector(theta, phi, N_left, angle)
        N_right = roArrayVector(theta, phi, N_right, angle)
        N_up = roArrayVector(theta, phi, N_up, angle)
        N_down = roArrayVector(theta, phi, N_down, angle)

        V = torch.stack([
            torch.cross(N_left, N_up), torch.cross(N_down, N_left),
            torch.cross(N_up, N_right), torch.cross(N_right, N_down)
        ])

        norm = torch.norm(V, dim=2, keepdim=True)
        V = V / norm

        # Fixing the E array formation
        E = torch.cat([
            torch.cat([N_left.unsqueeze(2), N_up.unsqueeze(2)], dim=2),
            torch.cat([N_down.unsqueeze(2), N_left.unsqueeze(2)], dim=2),
            torch.cat([N_up.unsqueeze(2), N_right.unsqueeze(2)], dim=2),
            torch.cat([N_right.unsqueeze(2), N_down.unsqueeze(2)], dim=2)
        ], dim=1)

        return torch.stack([N_left, N_right, N_up, N_down]), V, E.view(-1, 4, 3)
    
    def interArea(self, orders, E):
        '''Intersection Area Computation'''
        angles = -torch.matmul(E[:, 0, :].unsqueeze(1), E[:, 1, :].unsqueeze(2))
        angles = torch.clamp(angles, -1, 1)
        whole_inter = torch.arccos(angles)
        inter_res = torch.zeros(orders.shape[0], device=orders.device)
        loop = 0
        idx = torch.where(orders != 0)[0]
        iters = orders[idx]
        for i, j in enumerate(iters):
            inter_res[idx[i]] = torch.sum(
                whole_inter[loop:loop+j], dim=0) - (j - 2) * torch.pi
            loop += j
        return inter_res

    def remove_outer_points(self, dets, gt):
        '''Remove points outside the two spherical rectangles'''
        N_dets, V_dets, E_dets = self.getNormal(dets)
        N_gt, V_gt, E_gt = self.getNormal(gt)
        N_res = torch.cat((N_dets, N_gt), dim=0)
        V_res = torch.cat((V_dets, V_gt), dim=0)

        N_dets_expand = N_dets.repeat_interleave(N_gt.shape[0], dim=0)
        N_gt_expand = N_gt.repeat(N_dets.shape[0], 1, 1)

        tmp1 = torch.cross(N_dets_expand, N_gt_expand)
        mul1 = tmp1 / (torch.norm(tmp1, dim=2).unsqueeze(2).repeat(1, 1, tmp1.shape[2]) + 1e-10)

        tmp2 = torch.cross(N_gt_expand, N_dets_expand)
        mul2 = tmp2 / (torch.norm(tmp2, dim=2).unsqueeze(2).repeat(1, 1, tmp2.shape[2]) + 1e-10)

        V_res = torch.cat((V_res, mul1, mul2), dim=0)

        dimE = (E_res.shape[0] * 2, E_res.shape[1], E_res.shape[2], E_res.shape[3])
        E_res = torch.cat(
            (E_res, torch.cat((N_dets_expand, N_gt_expand), dim=1).view(dimE)), dim=0)
        E_res = torch.cat(
            (E_res, torch.cat((N_gt_expand, N_dets_expand), dim=1).view(dimE)), dim=0)

        res = torch.round(torch.matmul(V_res.transpose(1, 0, 2), N_res.transpose(1, 2, 0)), decimals=8)
        value = torch.all(res >= 0, dim=2)
        return value, V_res, E_res

    def computeInter(self, dets, gt):
        '''The whole Intersection Area Computation Process'''
        value, V_res, E_res = self.remove_outer_points(dets, gt)

        ind0 = torch.where(value)[0]
        ind1 = torch.where(value)[1]

        if ind0.shape[0] == 0:
            return torch.zeros((dets.shape[0]), device=dets.device)

        E_final = E_res[ind1, :, ind0, :]
        orders = torch.bincount(ind0)

        minus = dets.shape[0] - orders.shape[0]
        if minus > 0:
            orders = torch.nn.functional.pad(orders, (0, minus), mode='constant')

        inter = self.interArea(orders, E_final)
        return inter

    def sphIoU(self, dets, gt):
        '''Unbiased Spherical IoU Computation'''
        if not dets.numel() or not gt.numel():
            return torch.zeros(dets.shape[0], device=dets.device)

        d_size, g_size = dets.shape[0], gt.shape[0]
        res = torch.cat((dets.repeat(g_size, 1), gt.repeat(d_size, 1, 1).view(-1, gt.shape[1])), dim=1)
        area_A = self.area(res[:, 2], res[:, 3])
        area_B = self.area(res[:, 7], res[:, 8])
        inter = self.computeInter(res[:, :5], res[:, 5:])
        final = (inter / (area_A + area_B - inter)).view(d_size, g_size)
        return final