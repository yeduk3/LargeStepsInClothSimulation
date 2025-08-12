//
//  main.cpp
//  LargeStepInClothSimulationXCode
//
//  Created by 이용규 on 8/4/25.
//
#include <YGLWindow.hpp>
#include <camera.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include <list>
#include <program.hpp>
#include <sys/types.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

//#define STB_IMAGE_IMPLEMENTATION
//#include <stb_image.h>

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);


using Eigen::SparseMatrix;
using Eigen::VectorXf;
using Eigen::Vector3f;
using Eigen::Triplet;
using Eigen::Matrix3f;

SparseMatrix<float> M, A, P, invP;
VectorXf b, r;
std::vector<Matrix3f> S;

Program shader;

YGLWindow *window;

GLuint fabricTex;


VectorXf filter(const VectorXf &a) {
    VectorXf ret(a.size());
    for (auto i = 0; i < a.size()/3; i++) {
        ret.segment(i*3, 3) = S[i] * a.segment(i*3, 3);
    }
    return ret;
}


VectorXf modified_pcg(const VectorXf &z, const float ep2 = 0.1f) {
    auto dv = z;
    auto bf = filter(b);
    auto d0 = bf.dot(P * bf);
    r = filter(b - A * dv);
    auto c = filter(invP * r);
    auto dn = r.dot(c);
    
    const int MAX_ITER = 100;
    int iter = 0;
    while (dn > ep2 * d0) {
        if(iter++ >= MAX_ITER) {
            printf("Warning! Modified PCG Not Converged! - Max Iter\n");
            break;
        }
        auto q = filter(A * c);
        auto cdotq = c.dot(q);
        auto al = dn / cdotq;
        dv = dv + al * c;
        r = r - al * q;
        auto s = invP * r;
        auto dold = dn;
        dn = r.dot(s);
        c = filter(s + dn / dold * c);
    }
    
    if(!(dv.array().isFinite().all())) {
        printf("Warning! Modified PCG's result contain NaN");
        dv.setZero();
    }

    return dv;
}

struct Plane {
    Vector3f v[6];
    Plane(Vector3f p, float s = 1) {
        v[0] = p+Vector3f(-s / 2, 0, -s / 2);
        v[1] = p+Vector3f(-s / 2, 0, s / 2);
        v[2] = p+Vector3f(s / 2, 0, s / 2);
        v[3] = p+Vector3f(-s / 2, 0, -s / 2);
        v[4] = p+Vector3f(s / 2, 0, s / 2);
        v[5] = p+Vector3f(s / 2, 0, -s / 2);
    }
    GLuint vao = -1, vbo;
    void draw() {
        if (vao == -1) {
            glGenVertexArrays(1, &vao);
            glBindVertexArray(vao);

            glGenBuffers(1, &vbo);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(Vector3f), v, GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3f), 0);
        }
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
};
Plane p({0, 0, 0}, 100);

struct Cloth {

    VectorXf x, v; // in R^3 domain
    VectorXf uv;   // in R^2 domain

    // per triangle or per adj triangles
    // ccw
    std::vector<Eigen::Vector3i> face; // 16 bit unsigned
    // triangle_strip
    std::vector<Eigen::Vector4i> adjface;

    GLuint vao = -1, vbo, veo, vto;
    void draw() {
        if (vao == -1) {
            glGenVertexArrays(1, &vao);
            glGenBuffers(1, &vbo);
            glGenBuffers(1, &vto);
            glGenBuffers(1, &veo);
            std::cout << "Cloth Buffer Generated" << std::endl;
        }
        glBindVertexArray(vao);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, x.size() * sizeof(float), x.data(), GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
        
        glBindBuffer(GL_ARRAY_BUFFER, vto);
        glBufferData(GL_ARRAY_BUFFER, uv.size() * sizeof(float), uv.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), 0);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, veo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, face.size() * sizeof(Eigen::Vector3i), face.data(), GL_STATIC_DRAW);
        
        shader.setTexture("fabricTex", 0, fabricTex);
        
        glDrawElements(GL_TRIANGLES, int(face.size() * 3), GL_UNSIGNED_INT, 0);
    }
};
Cloth c;

const int NGRID = 20;
const int NPARTICLES = NGRID * NGRID;

const float stretch_stiff = 1000;
const float shear_stiff = 100;
const float bend_stiff = 50;
const float damping = 5;

VectorXf f0(NPARTICLES*3), z(NPARTICLES*3);
SparseMatrix<float> dfdv(NPARTICLES*3, NPARTICLES*3), dfdx(NPARTICLES*3, NPARTICLES*3);



void addBlockToTriplets(std::vector<Triplet<float>>& triplets, int r_idx, int c_idx, const Matrix3f& block) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (block(i, j) != 0.0f) {
                triplets.emplace_back(3 * r_idx + i, 3 * c_idx + j, block(i, j));
            }
        }
    }
}

void clearForce() {
    dfdv.setZero();
    dfdx.setZero();
    f0.setZero();
}

void computeStretch(const Eigen::Vector3i &face, std::vector<Triplet<float>>& triplets, std::vector<Triplet<float>>& v_triplets, float bu = 1, float bv = 1) {
    auto i = face.x();
    auto j = face.y();
    auto k = face.z();

    auto dx1 = c.x.segment(j*3, 3) - c.x.segment(i*3, 3);
    auto dx2 = c.x.segment(k*3, 3) - c.x.segment(i*3, 3);
    auto du1 = c.uv(j*2)   - c.uv(i*2);
    auto du2 = c.uv(k*2)   - c.uv(i*2);
    auto dv1 = c.uv(j*2+1) - c.uv(i*2+1);
    auto dv2 = c.uv(k*2+1) - c.uv(i*2+1);

    auto detdudv = du1 * dv2 - du2 * dv1;
    if(std::abs(detdudv) < 1E-9) {
        printf("Warning! The uv coord is too compact!\n");
        return;
    }
    auto wu = (dx1 * dv2 - dx2 * dv1) / detdudv;
    auto wv = (-dx1 * du2 + dx2 * du1) / detdudv;
    float lwu = wu.norm();
    float lwv = wv.norm();
    
    if(lwu < 1E-9 || lwv < 1E-9) return;
    
    Vector3f nwu = wu/lwu;
    Vector3f nwv = wv/lwv;

    auto a = abs(detdudv) * 0.5f;

    Vector3f Cui, Cuj, Cuk, Cvi, Cvj, Cvk;
    auto dwui = (-dv2 + dv1) / detdudv;
    auto dwuj = dv2 / detdudv;
    auto dwuk = -dv1 / detdudv;
    auto dwvi = (du2 - du1) / detdudv;
    auto dwvj = -du2 / detdudv;
    auto dwvk = du1 / detdudv;
    Cui = a * dwui * nwu;
    Cuj = a * dwuj * nwu;
    Cuk = a * dwuk * nwu;
    Cvi = a * dwvi * nwv;
    Cvj = a * dwvj * nwv;
    Cvk = a * dwvk * nwv;

    Matrix3f Cuii, Cujj, Cukk, Cuij, Cuik, Cujk;
    Matrix3f M_nwu; M_nwu.col(0) = nwu; M_nwu.col(1) = nwu; M_nwu.col(2) = nwu;
    Matrix3f M_nwv; M_nwv.col(0) = nwv; M_nwv.col(1) = nwv; M_nwv.col(2) = nwv;
//    Matrix3f u_term = (1.f/lwu)*Matrix3f::Identity() - (1.f/(lwu*lwu))*M_nwu*wu.asDiagonal();
//    Matrix3f v_term = (1.f/lwv)*Matrix3f::Identity() - (1.f/(lwv*lwv))*M_nwv*wv.asDiagonal();
    Matrix3f u_term = (1.f/lwu)*Matrix3f::Identity() - (1.f/(lwu*lwu))*nwu*wu.transpose();
    Matrix3f v_term = (1.f/lwv)*Matrix3f::Identity() - (1.f/(lwv*lwv))*nwv*wv.transpose();
    Cuii = a * dwui * dwui * u_term;
    Cujj = a * dwuj * dwuj * u_term;
    Cukk = a * dwuk * dwuk * u_term;
    Cuij = a * dwui * dwuj * u_term;
    Cuik = a * dwui * dwuk * u_term;
    Cujk = a * dwuj * dwuk * u_term;

    Matrix3f Cvii, Cvjj, Cvkk, Cvij, Cvik, Cvjk;
    Cvii = a * dwvi * dwvi * v_term;
    Cvjj = a * dwvj * dwvj * v_term;
    Cvkk = a * dwvk * dwvk * v_term;
    Cvij = a * dwvi * dwvj * v_term;
    Cvik = a * dwvi * dwvk * v_term;
    Cvjk = a * dwvj * dwvk * v_term;

    Matrix3f Kii, Kjj, Kkk, Kij, Kik, Kjk;

    float Cu = a * (lwu - bu);
    float Cv = a * (lwv - bv);
    
    float dampui = Cui.dot(c.v.segment(i*3, 3));
    float dampvi = Cvi.dot(c.v.segment(i*3, 3));
    float dampuj = Cuj.dot(c.v.segment(j*3, 3));
    float dampvj = Cvj.dot(c.v.segment(j*3, 3));
    float dampuk = Cuk.dot(c.v.segment(k*3, 3));
    float dampvk = Cvk.dot(c.v.segment(k*3, 3));

    Kii = -stretch_stiff * ((Cui * Cui.transpose()) + (Cvi * Cvi.transpose()) + Cuii * Cu + Cvii * Cv) - damping*(Cuii*dampui+Cvii*dampvi);
    Kjj = -stretch_stiff * ((Cuj * Cuj.transpose()) + (Cvj * Cvj.transpose()) + Cujj * Cu + Cvjj * Cv) - damping*(Cujj*dampuj+Cvjj*dampvj);
    Kkk = -stretch_stiff * ((Cuk * Cuk.transpose()) + (Cvk * Cvk.transpose()) + Cukk * Cu + Cvkk * Cv) - damping*(Cukk*dampuk+Cvkk*dampvk);
    Kij = -stretch_stiff * ((Cui * Cuj.transpose()) + (Cvi * Cvj.transpose()) + Cuij * Cu + Cvij * Cv) - damping*(Cuij*dampui+Cvij*dampvi);
    Kik = -stretch_stiff * ((Cui * Cuk.transpose()) + (Cvi * Cvk.transpose()) + Cuik * Cu + Cvik * Cv) - damping*(Cuik*dampui+Cvik*dampvi);
    Kjk = -stretch_stiff * ((Cuj * Cuk.transpose()) + (Cvj * Cvk.transpose()) + Cujk * Cu + Cvjk * Cv) - damping*(Cujk*dampuj+Cvjk*dampvj);
    
    addBlockToTriplets(triplets, i, i, Kii);
    addBlockToTriplets(triplets, j, j, Kjj);
    addBlockToTriplets(triplets, k, k, Kkk);
    addBlockToTriplets(triplets, i, j, Kij);
    addBlockToTriplets(triplets, j, i, Kij.transpose());
    addBlockToTriplets(triplets, i, k, Kik);
    addBlockToTriplets(triplets, k, i, Kik.transpose());
    addBlockToTriplets(triplets, j, k, Kjk);
    addBlockToTriplets(triplets, k, j, Kjk.transpose());
    
    addBlockToTriplets(v_triplets, i, i,  -damping*(Cui*Cui.transpose()+Cvi*Cvi.transpose()));
    addBlockToTriplets(v_triplets, j, j,  -damping*(Cuj*Cuj.transpose()+Cvj*Cvj.transpose()));
    addBlockToTriplets(v_triplets, k, k,  -damping*(Cuk*Cuk.transpose()+Cvk*Cvk.transpose()));
    addBlockToTriplets(v_triplets, i, j,  -damping*(Cui*Cuj.transpose()+Cvi*Cvj.transpose()));
    addBlockToTriplets(v_triplets, j, i, (-damping*(Cui*Cuj.transpose()+Cvi*Cvj.transpose()).transpose()));
    addBlockToTriplets(v_triplets, i, k,  -damping*(Cui*Cuk.transpose()+Cvi*Cvk.transpose()));
    addBlockToTriplets(v_triplets, k, i, (-damping*(Cui*Cuk.transpose()+Cvi*Cvk.transpose()).transpose()));
    addBlockToTriplets(v_triplets, j, k,  -damping*(Cuj*Cuk.transpose()+Cvj*Cvk.transpose()));
    addBlockToTriplets(v_triplets, k, j, (-damping*(Cuj*Cuk.transpose()+Cvj*Cvk.transpose())).transpose());
    
    f0.segment(i*3, 3) += -stretch_stiff * (Cui*Cu + Cvi*Cv) - damping*(Cui*dampui+Cvi*dampvi);
    f0.segment(j*3, 3) += -stretch_stiff * (Cuj*Cu + Cvj*Cv) - damping*(Cuj*dampuj+Cvj*dampvj);
    f0.segment(k*3, 3) += -stretch_stiff * (Cuk*Cu + Cvk*Cv) - damping*(Cuk*dampuk+Cvk*dampvk);
    
    
    
    if(!(f0.array().isFinite().all())) {
        std::cerr << "Stretch Error: Vector 'f0' contains NaN or Inf!" << std::endl;
        f0.setZero();
        return;
    }
}

void computeShear(const Eigen::Vector3i &face, std::vector<Triplet<float>>& triplets) {
    auto i = face.x();
    auto j = face.y();
    auto k = face.z();

    auto dx1 = c.x.segment(j*3, 3) - c.x.segment(i*3, 3);
    auto dx2 = c.x.segment(k*3, 3) - c.x.segment(i*3, 3);
    auto du1 = c.uv(j*2)   - c.uv(i*2);
    auto du2 = c.uv(k*2)   - c.uv(i*2);
    auto dv1 = c.uv(j*2+1) - c.uv(i*2+1);
    auto dv2 = c.uv(k*2+1) - c.uv(i*2+1);

    auto detdudv = du1 * dv2 - du2 * dv1;
    if(std::abs(detdudv) < 1E-9) {
        printf("Warning! The uv coord is too compact!\n");
        return;
    }
    auto wu = (dx1 * dv2 - dx2 * dv1) / detdudv;
    auto wv = (-dx1 * du2 + dx2 * du1) / detdudv;
    float lwu = wu.norm();
    float lwv = wv.norm();
    
    if(lwu < 1E-9 || lwv < 1E-9) return;
    
    Vector3f nwu = wu/lwu;
    Vector3f nwv = wv/lwv;

    auto a = abs(detdudv) * 0.5f;

    Vector3f Ci, Cj, Ck;
    auto dwui = (-dv2 + dv1) / detdudv;
    auto dwuj = dv2 / detdudv;
    auto dwuk = -dv1 / detdudv;
    auto dwvi = (du2 - du1) / detdudv;
    auto dwvj = -du2 / detdudv;
    auto dwvk = du1 / detdudv;
    Ci = a * (dwui*wv + dwvi*wu);
    Cj = a * (dwuj*wv + dwvj*wu);
    Ck = a * (dwuk*wv + dwvk*wu);


    Matrix3f Cii, Cjj, Ckk, Cij, Cik, Cjk;
    Cii = 2 * a * dwui * dwvi * Matrix3f::Identity();
    Cjj = 2 * a * dwuj * dwvj * Matrix3f::Identity();
    Ckk = 2 * a * dwuk * dwvk * Matrix3f::Identity();
    Cij = a * (dwui*dwvj + dwvi*dwuj) * Matrix3f::Identity();
    Cik = a * (dwui*dwvk + dwvi*dwuk) * Matrix3f::Identity();
    Cjk = a * (dwuj*dwvk + dwvj*dwuk) * Matrix3f::Identity();

    Matrix3f Kii, Kjj, Kkk, Kij, Kik, Kjk;

    float C = a * wu.dot(wv);
    
    Kii = -shear_stiff * (Ci*Ci.transpose() + Cii*C);
    Kjj = -shear_stiff * (Cj*Cj.transpose() + Cjj*C);
    Kkk = -shear_stiff * (Ck*Ck.transpose() + Ckk*C);
    Kij = -shear_stiff * (Ci*Cj.transpose() + Cij*C);
    Kik = -shear_stiff * (Ci*Ck.transpose() + Cik*C);
    Kjk = -shear_stiff * (Cj*Ck.transpose() + Cjk*C);
    
    addBlockToTriplets(triplets, i, i, Kii);
    addBlockToTriplets(triplets, j, j, Kjj);
    addBlockToTriplets(triplets, k, k, Kkk);
    addBlockToTriplets(triplets, i, j, Kij);
    addBlockToTriplets(triplets, j, i, Kij.transpose());
    addBlockToTriplets(triplets, i, k, Kik);
    addBlockToTriplets(triplets, k, i, Kik.transpose());
    addBlockToTriplets(triplets, j, k, Kjk);
    addBlockToTriplets(triplets, k, j, Kjk.transpose());
    
    f0.segment(i*3, 3) += -shear_stiff * (Ci*C);
    f0.segment(j*3, 3) += -shear_stiff * (Cj*C);
    f0.segment(k*3, 3) += -shear_stiff * (Ck*C);
    
    
    
    if(!(f0.array().isFinite().all())) {
        std::cerr << "Shear Error: Vector 'f0' contains NaN or Inf!" << std::endl;
        f0.setZero();
        return;
    }
}

Matrix3f derivMatrix(const Vector3f& v) {
    Matrix3f m;
    m <<     0, -v.z(),  v.y(),
          v.z(),      0, -v.x(),
         -v.y(),  v.x(),      0;
    return m;
}

void computeBend(const Eigen::Vector4i &adjFace, std::vector<Triplet<float>>& triplets) {
    
    //   x(k) - z(i)
    //    |   /   |
    //   y(j) - w(p)
    auto k = adjFace.x();
    auto j = adjFace.y();
    auto i = adjFace.z();
    auto p = adjFace.w();
    
    auto xk = c.x.segment(k*3, 3);
    auto xj = c.x.segment(j*3, 3);
    auto xi = c.x.segment(i*3, 3);
    auto xp = c.x.segment(p*3, 3);
    
    auto ik = xk - xi;
    auto ij = xj - xi;
    auto ip = xp - xi;
    
    Vector3f N1(ik.y()*ij.z() - ik.z()*ij.y(), -(ik.x()*ij.z() - ik.z()*ij.x()), ik.x()*ij.y() - ik.y()*ij.x());
    Vector3f N2(ij.y()*ip.z() - ij.z()*ip.y(), -(ij.x()*ip.z() - ij.z()*ip.x()), ij.x()*ip.y() - ij.y()*ip.x());
    float lN1 = N1.norm();
    float lN2 = N2.norm();
    float lij = ij.norm();
    
    if(lN1 < 1E-9 || lN2 < 1E-9) return;
    
    auto n1 = N1/lN1;
    auto n2 = N2/lN2;
    auto ce = ij/lij;
    
    auto invLN1 = 1.f/lN1;
    auto invLN2 = 1.f/lN2;
    
    float cosTheta = fmax(-1.f, fmin(1.f, n1.dot(n2)));
//    float sinTheta = fmax(-1.f, fmin(1.f, n1.cross(n2).dot(ce)));
    float sinTheta2 = 1.f/(1-cosTheta*cosTheta);
    if(sinTheta2 < 1E-9) return;
    float sinTheta = fmax(-1.f, fmin(1.f, sqrt(sinTheta2)));
//    float sinTheta = sqrt(1.f/(1-cosTheta*cosTheta));
    
    // dn1i = derivative of n1 w.r.t. xi
    Matrix3f dn1i, dn1j, dn1k, dn1p, dn2i, dn2j, dn2k, dn2p;
    dn1i = invLN1 * derivMatrix(-ij+ik);
    dn1j = invLN1 * derivMatrix(-ik);
    dn1k = invLN1 * derivMatrix(ij);
    dn1p = Matrix3f::Zero();
    dn2i = invLN2 * derivMatrix(-ip+ij);
    dn2j = invLN2 * derivMatrix(ip);
    dn2k = Matrix3f::Zero();
    dn2p = invLN2 * derivMatrix(-ij);
    
    
    
    auto invSinTheta = 1.f/sinTheta;
    if(abs(sinTheta) < 1E-9 || abs(cosTheta) > 1-(1E-9)) {
//        printf("THETA SMALL\n");
        return;
    }
    
    Vector3f ri = dn1i*n2 + dn2i*n1;
    Vector3f rj = dn1j*n2 + dn2j*n1;
    Vector3f rk = dn1k*n2 + dn2k*n1;
    Vector3f rp = dn1p*n2 + dn2p*n1;
    
    Vector3f dti = -invSinTheta * ri;
    Vector3f dtj = -invSinTheta * rj;
    Vector3f dtk = -invSinTheta * rk;
    Vector3f dtp = -invSinTheta * rp;
    
//    Matrix3f dn1ii_n2 = Matrix3f::Zero();
//    Matrix3f dn1ij_n2 = invLN1 * invLN2 * derivMatrix(-n2);
//    Matrix3f dn1ik_n2 = invLN1 * invLN2 * derivMatrix(n2);
//    Matrix3f dn1ip_n2 = Matrix3f::Zero();
//    
//    Matrix3f dn1ji_n2 = invLN1 * invLN2 * derivMatrix(n2);
//    Matrix3f dn1jj_n2 = Matrix3f::Zero();
//    Matrix3f dn1jk_n2 = invLN1 * invLN2 * derivMatrix(-n2);
//    Matrix3f dn1jp_n2 = Matrix3f::Zero();
//    
//    Matrix3f dn1ki_n2 = invLN1 * invLN2 * derivMatrix(-n2);
//    Matrix3f dn1kj_n2 = invLN1 * invLN2 * derivMatrix(n2);
//    Matrix3f dn1kk_n2 = Matrix3f::Zero();
//    Matrix3f dn1kp_n2 = Matrix3f::Zero();
//    
//    Matrix3f dn1pi_n2 = Matrix3f::Zero();
//    Matrix3f dn1pj_n2 = Matrix3f::Zero();
//    Matrix3f dn1pk_n2 = Matrix3f::Zero();
//    Matrix3f dn1pp_n2 = Matrix3f::Zero();
//    
//    Matrix3f dn2ii_n1 = Matrix3f::Zero();
//    Matrix3f dn2ij_n1 = invLN1 * invLN2 * derivMatrix(n1);
//    Matrix3f dn2ik_n1 = Matrix3f::Zero();
//    Matrix3f dn2ip_n1 = invLN1 * invLN2 * derivMatrix(-n1);
//    
//    Matrix3f dn2ji_n1 = invLN1 * invLN2 * derivMatrix(-n1);
//    Matrix3f dn2jj_n1 = Matrix3f::Zero();
//    Matrix3f dn2jk_n1 = Matrix3f::Zero();
//    Matrix3f dn2jp_n1 = invLN1 * invLN2 * derivMatrix(n1);
//    
//    Matrix3f dn2ki_n1 = Matrix3f::Zero();
//    Matrix3f dn2kj_n1 = Matrix3f::Zero();
//    Matrix3f dn2kk_n1 = Matrix3f::Zero();
//    Matrix3f dn2kp_n1 = Matrix3f::Zero();
//    
//    Matrix3f dn2pi_n1 = invLN1 * invLN2 * derivMatrix(n1);
//    Matrix3f dn2pj_n1 = invLN1 * invLN2 * derivMatrix(-n1);
//    Matrix3f dn2pk_n1 = Matrix3f::Zero();
//    Matrix3f dn2pp_n1 = Matrix3f::Zero();
//    
//    auto cs2 = cosTheta/(sinTheta*sinTheta);
//    if(cs2 < 1E-9) return;
//    Matrix3f dtii = -invSinTheta*(dn1ii_n2+dn2ii_n1) + (dti*cs2)*ri.transpose();
//    Matrix3f dtij = -invSinTheta*(dn1ij_n2+dn2ij_n1) + (dtj*cs2)*ri.transpose();
//    Matrix3f dtik = -invSinTheta*(dn1ik_n2+dn2ik_n1) + (dtk*cs2)*ri.transpose();
//    Matrix3f dtip = -invSinTheta*(dn1ip_n2+dn2ip_n1) + (dtp*cs2)*ri.transpose();
//    Matrix3f dtjj = -invSinTheta*(dn1jj_n2+dn2jj_n1) + (dtj*cs2)*rj.transpose();
//    Matrix3f dtjk = -invSinTheta*(dn1jk_n2+dn2jk_n1) + (dtk*cs2)*rj.transpose();
//    Matrix3f dtjp = -invSinTheta*(dn1jp_n2+dn2jp_n1) + (dtp*cs2)*rj.transpose();
//    Matrix3f dtkk = -invSinTheta*(dn1kk_n2+dn2kk_n1) + (dtk*cs2)*rk.transpose();
//    Matrix3f dtkp = -invSinTheta*(dn1kp_n2+dn2kp_n1) + (dtp*cs2)*rk.transpose();
//    Matrix3f dtpp = -invSinTheta*(dn1pp_n2+dn2pp_n1) + (dtp*cs2)*rp.transpose();
    
    
    
    float theta = acos(cosTheta);
    
    
//    Matrix3f Kii = -stiff * (dti*dti.transpose() + dtii*theta);
//    Matrix3f Kij = -stiff * (dti*dtj.transpose() + dtij*theta);
//    Matrix3f Kik = -stiff * (dti*dtk.transpose() + dtik*theta);
//    Matrix3f Kip = -stiff * (dti*dtp.transpose() + dtip*theta);
//    Matrix3f Kjj = -stiff * (dtj*dtj.transpose() + dtjj*theta);
//    Matrix3f Kjk = -stiff * (dtj*dtk.transpose() + dtjk*theta);
//    Matrix3f Kjp = -stiff * (dtj*dtp.transpose() + dtjp*theta);
//    Matrix3f Kkk = -stiff * (dtk*dtk.transpose() + dtkk*theta);
//    Matrix3f Kkp = -stiff * (dtk*dtp.transpose() + dtkp*theta);
//    Matrix3f Kpp = -stiff * (dtp*dtp.transpose() + dtpp*theta);
    
    // Gauss-Newton Approximation???????
    Matrix3f Kii = -bend_stiff * (dti*dti.transpose());
    Matrix3f Kij = -bend_stiff * (dti*dtj.transpose());
    Matrix3f Kik = -bend_stiff * (dti*dtk.transpose());
    Matrix3f Kip = -bend_stiff * (dti*dtp.transpose());
    Matrix3f Kjj = -bend_stiff * (dtj*dtj.transpose());
    Matrix3f Kjk = -bend_stiff * (dtj*dtk.transpose());
    Matrix3f Kjp = -bend_stiff * (dtj*dtp.transpose());
    Matrix3f Kkk = -bend_stiff * (dtk*dtk.transpose());
    Matrix3f Kkp = -bend_stiff * (dtk*dtp.transpose());
    Matrix3f Kpp = -bend_stiff * (dtp*dtp.transpose());
    
    
    addBlockToTriplets(triplets, i, i, Kii);
    addBlockToTriplets(triplets, i, j, Kij);
    addBlockToTriplets(triplets, i, k, Kik);
    addBlockToTriplets(triplets, i, p, Kip);
    addBlockToTriplets(triplets, j, i, Kij.transpose());
    addBlockToTriplets(triplets, j, j, Kjj);
    addBlockToTriplets(triplets, j, k, Kjk);
    addBlockToTriplets(triplets, j, p, Kjp);
    addBlockToTriplets(triplets, k, i, Kik.transpose());
    addBlockToTriplets(triplets, k, j, Kjk.transpose());
    addBlockToTriplets(triplets, k, k, Kkk);
    addBlockToTriplets(triplets, k, p, Kkp);
    addBlockToTriplets(triplets, p, i, Kip.transpose());
    addBlockToTriplets(triplets, p, j, Kjp.transpose());
    addBlockToTriplets(triplets, p, k, Kkp.transpose());
    addBlockToTriplets(triplets, p, p, Kpp);
    
    f0.segment(i*3, 3) += -bend_stiff * (dti*theta);
    f0.segment(j*3, 3) += -bend_stiff * (dtj*theta);
    f0.segment(k*3, 3) += -bend_stiff * (dtk*theta);
    f0.segment(p*3, 3) += -bend_stiff * (dtp*theta);
    
    
    if(!(f0.array().isFinite().all())) {
        std::cerr << "Bend Error: Vector 'f0' contains NaN or Inf!" << std::endl;
        f0.setZero();
        return;
    }
}


void resolveCollision() {
    // particles
    S = std::vector<Matrix3f>(NPARTICLES, Matrix3f::Identity());
    const float ep = 1E-6;
    for(int i = 0; i < NPARTICLES; i++) {
        auto X = c.x.segment(i*3, 3);
        auto V = c.v.segment(i*3, 3);
        // plane
        for(int t = 0; t < 2; t++) {
            auto P = Vector3f(p.v[t*3].x(), p.v[t*3].y(), p.v[t*3].z());
            auto N = Vector3f(0, 1, 0);
            if((X-P).dot(N) < ep && N.dot(V) < 0) {
                S[i] = S[i] - (-N)*(-N).transpose();

                printf("Collision!!!!\n");
            }
        }
    }
}

float randf() {
    return rand() / float(RAND_MAX);
}


unsigned int loadTexture() {
    stbi_set_flip_vertically_on_load(true);

    int w, h, comp;
    unsigned char *data = stbi_load("fabric_pattern_05_col_01_4k.png", &w, &h, &comp, 0);
    
    unsigned int textureID;
    glGenTextures(1, &textureID);
    
    if(data) {
        GLenum format = GL_RGBA;
        if (comp == 1)
            format = GL_RED;
        else if (comp == 3)
            format = GL_RGB;
        else if (comp == 4)
            format = GL_RGBA;
        glBindTexture(GL_TEXTURE_2D, textureID); // 텍스처 바인딩
        
        // 로드한 이미지 데이터로 텍스처를 생성합니다.
        glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D); // 밉맵 자동 생성
        
        // 텍스처 파라미터 설정
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        // 이미지 데이터를 메모리에서 해제합니다.
        stbi_image_free(data);

    } else {
        printf("Texture load failed.\n");
    }
    return textureID;
}

void createCloth() {
    // create cloth mesh
    c.x  = VectorXf::Zero(NPARTICLES*3);
    c.v  = VectorXf::Zero(NPARTICLES*3);
    c.uv = VectorXf::Zero(NPARTICLES*2);
    //  0  1  2 ... 19
    // 20 21 22 ... 39
    for (int row = 0; row < NGRID; row++) {
        for (int col = 0; col < NGRID; col++) {
            auto idx = row * NGRID + col;
            auto pos = Vector3f((col - 9.5) * 1, (9.5 - row) * 1, 0); // centimeter?
            c.x.segment(idx*3, 3)  = Vector3f(pos.x(), 20, -pos.y()+9.5);
            c.uv.segment(idx*2, 2) = pos.segment(0, 2);
        }
    }
//    c.x.segment(0, 3) = Vector3f(-15, 15, 0);
    c.face.clear();
    c.adjface.clear();
    for (int row = 0; row < NGRID - 1; row++) {
        for (int col = 0; col < NGRID - 1; col++) {
            int lt = row * NGRID + col;
            c.face.push_back({lt, lt + NGRID, lt + 1});
            c.face.push_back({lt + 1, lt + NGRID, lt + NGRID + 1});
            c.adjface.push_back({lt, lt + NGRID, lt + 1, lt + NGRID + 1});
            if(row > 0 && col < NGRID-2) c.adjface.push_back({lt, lt+1, lt-NGRID+1, lt-NGRID+2});
            if(col > 0 && row < NGRID-2) c.adjface.push_back({lt, lt+NGRID-1, lt+NGRID, lt+NGRID*2-1});
        }
    }
    std::cout << "Face Generated: " << c.face.size() << std::endl;
    std::cout << "Adj Face Generated: " << c.adjface.size() << std::endl;

    // Set not changing matrix
    M = SparseMatrix<float>(NPARTICLES*3, NPARTICLES*3); // 1 kg?
    std::vector<Triplet<float>> M_tri;
    for (int i = 0; i < NPARTICLES*3; i++) M_tri.emplace_back(i, i, 1);
    M.setFromTriplets(M_tri.begin(), M_tri.end());
    S = std::vector<Matrix3f>(NPARTICLES, Matrix3f::Identity());
    S[0] = Matrix3f::Zero(3, 3);
//    S[NGRID-1] = Matrix3f::Zero(3, 3);
    f0 = VectorXf::Zero(NPARTICLES*3);
    z = VectorXf::Zero(NPARTICLES*3);
}

void init() {
    camera.setPosition({0, 0, 50});
    camera.glfwSetCallbacks(window->getGLFWWindow());
    glfwSetKeyCallback(window->getGLFWWindow(), key_callback);

    shader.loadShader("cloth.vert", "cloth.geom", "cloth.frag");

    createCloth();
    fabricTex = loadTexture();
}


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_0 && action == GLFW_PRESS) {
        createCloth();
    }
}

void render() {
    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, window->width(), window->height());
    glClearColor(0, 0, 0.3, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set matrix
    clearForce();
    
    std::vector<Triplet<float>> K_tri, V_tri;
    for (auto &f : c.face) {
        computeStretch(f, K_tri, V_tri);
        computeShear(f, K_tri);
    }
    for (auto &adjF : c.adjface) {
        computeBend(adjF, K_tri);
    }
    dfdx.setFromTriplets(K_tri.begin(), K_tri.end());
    dfdv.setFromTriplets(V_tri.begin(), V_tri.end());
    
    for (int i = 0; i < NPARTICLES; i++) {
        f0(i*3+1) += -9.8 * M.coeff(i*3+1, i*3+1);
//        std::cout <<"f0: "<<f0(i*3+1) << std::endl;
    }

    float h = 1 / 60.f;

    A = M - h * dfdv - h * h * dfdx;
    b = h * (f0 + h * dfdx * c.v);
    auto A_diag = A.diagonal();
    invP = A_diag.asDiagonal();
    P = A_diag.cwiseInverse().asDiagonal();

    // Modified pcg
//    z.segment(0, 3) = Vector3f(1,0,0);
    auto dv = modified_pcg(z);
    
//    resolveCollision();

    // Euler integration
    c.v = c.v + dv;
    c.x = c.x + h * c.v;

    // Draw
    shader.use();
    glm::mat4 ModelMat = glm::mat4(1);
    glm::mat4 ViewMat = camera.lookAt();
    glm::mat4 MV = ViewMat * ModelMat;
    glm::mat4 MVP = camera.perspective(window->aspect(), 0.1f, 1000.f) * MV;
    shader.setUniform("M", ModelMat);
    shader.setUniform("MVP", MVP);
    shader.setUniform("MV", MV);
    shader.setUniform("normalMat", glm::mat3(glm::vec3(MV[0]), glm::vec3(MV[1]),
                                             glm::vec3(MV[2])));
    shader.setUniform("viewMat", ViewMat);
    shader.setUniform("lightPosition", glm::vec3(10, 10, 5));
    shader.setUniform("lightColor", glm::vec3(160));
    shader.setUniform("diffuseColor", glm::vec3(1));
    shader.setUniform("specularColor", glm::vec3(0.33));
    shader.setUniform("shininess", 10.f);
    
//    p.draw();
    c.draw();
}



int main() {
    window = new YGLWindow(640, 480, "test");
    window->mainLoop(init, render);
    return 0;
}
