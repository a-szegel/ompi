/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2007 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013      Sandia National Laboratories.  All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2020      Amazon.com, Inc. or its affiliates.
 *                         All Rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "pml_cm.h"
#include "opal/util/event.h"
#include "ompi/mca/mtl/mtl.h"
#include "ompi/mca/mtl/base/base.h"
#include "ompi/mca/pml/base/pml_base_bsend.h"

#include "pml_cm_sendreq.h"
#include "pml_cm_recvreq.h"
#include "pml_cm_component.h"

static int mca_pml_cm_component_register(void);
static int mca_pml_cm_component_open(void);
static int mca_pml_cm_component_close(void);
static mca_pml_base_module_t* mca_pml_cm_component_init( int* priority,
                            bool enable_progress_threads, bool enable_mpi_threads);
static int mca_pml_cm_component_fini(void);

mca_pml_base_component_2_1_0_t mca_pml_cm_component = {
    /* First, the mca_base_component_t struct containing meta
     * information about the component itself */

    .pmlm_version = {
        MCA_PML_BASE_VERSION_2_1_0,

        .mca_component_name = "cm",
        MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                              OMPI_RELEASE_VERSION),
        .mca_open_component = mca_pml_cm_component_open,
        .mca_close_component = mca_pml_cm_component_close,
        .mca_register_component_params = mca_pml_cm_component_register,
    },
    .pmlm_data = {
        /* This component is not checkpoint ready */
        MCA_BASE_METADATA_PARAM_NONE
    },

    .pmlm_init = mca_pml_cm_component_init,
    .pmlm_finalize = mca_pml_cm_component_fini,
};

/* Array of send completion callback - one per send type
 * These are called internally by the library when the send
 * is completed from its perspective.
 */
void (*send_completion_callbacks[MCA_PML_BASE_SEND_SIZE])
     (struct mca_mtl_request_t *mtl_request) =
  { mca_pml_cm_send_request_completion,
    mca_pml_cm_send_request_completion,
    mca_pml_cm_send_request_completion,
    mca_pml_cm_send_request_completion,
    mca_pml_cm_send_request_completion } ;

static int
mca_pml_cm_component_register(void)
{

    ompi_pml_cm.free_list_num = 4;
    (void) mca_base_component_var_register(&mca_pml_cm_component.pmlm_version, "free_list_num",
                                           "Initial size of request free lists",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &ompi_pml_cm.free_list_num);

    ompi_pml_cm.free_list_max = -1;
    (void) mca_base_component_var_register(&mca_pml_cm_component.pmlm_version, "free_list_max",
                                           "Maximum size of request free lists",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &ompi_pml_cm.free_list_max);

    ompi_pml_cm.free_list_inc = 64;
    (void) mca_base_component_var_register(&mca_pml_cm_component.pmlm_version, "free_list_inc",
                                           "Number of elements to add when growing request free lists",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &ompi_pml_cm.free_list_inc);

    return OPAL_SUCCESS;
}


// Begin Testing Accelerator Framework!

#include "opal/mca/accelerator/accelerator.h"
#include "opal/mca/accelerator/base/base.h"
#include <cuda.h>
#include <cuda_runtime.h>



static int omb_get_local_rank(void);
static CUcontext setup_cuda(int dev_id);
static void cleanup_accelerator(void);
static void accelerator_test_cuda_not_configured(void);
static void accelerator_test_get_device(int dev_id);
static void accelerator_test_device_can_access_peer(void);
static void accelerator_test_malloc_checkaddr_free(int dev_id);
static void accelerator_test_register_memory(void);
static void accelerator_test_get_address_range(void);
static void accelerator_test_ipc_devices(void);
static void accelerator_test_memcpy(int dev_id);
static void accelerator_test_vector_memcpy(int dev_id);

static int omb_get_local_rank(void)
{
    char *str = NULL;
    int local_rank = -1;
    str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    local_rank = atoi(str);

    return local_rank;
}

static CUcontext setup_cuda(int dev_id) {
    CUresult curesult = CUDA_SUCCESS;
    CUdevice cuDevice;
    CUcontext cuContext;

    // SET UP CUDA
    cudaSetDevice(dev_id);
    curesult = cuInit(0);
    if (curesult != CUDA_SUCCESS) {
        printf("\n\n\nTEST FAILURE!!!! Failed to init cuda device\n\n\n");
    }

    curesult = cuDeviceGet(&cuDevice, dev_id);
    if (curesult != CUDA_SUCCESS) {
        printf("\n\n\nTEST FAILURE!!!! Failed to get cuda device\n\n\n");
    }

    curesult = cuDevicePrimaryCtxRetain(&cuContext, cuDevice);
    if (curesult != CUDA_SUCCESS) {
        printf("\n\n\nTEST FAILURE!!!! Failed to get cuda device context\n\n\n");
    }

    return cuContext;
}

static void setup_accelerator(void) {
    // Open Framework
    int ret = mca_base_framework_open(&opal_accelerator_base_framework, 0);
    if (OMPI_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! ACC Framework failed to open\n\n\n");
    }

    // Select CUDA Component
    if (OPAL_SUCCESS != (ret = opal_accelerator_base_select())) {
        printf("\n\n\nTEST FAILURE!!!! Failed selecting a component\n\n\n");
    }
}

static void cleanup_accelerator(void) {
    mca_base_framework_close(&opal_accelerator_base_framework);
}


static void cleanup_cuda(int dev_id) {
    cuDevicePrimaryCtxRelease ((CUdevice) dev_id);
    cuDevicePrimaryCtxReset((CUdevice) dev_id);
}

static void accelerator_test_cuda_not_configured(void) {
    int ret;
    void *ptr;

    // Attempt to Malloc without CUDA Context setup
    ret = opal_accelerator.malloc(&ptr, 16);
    if (OPAL_SUCCESS != ret) {
        printf("TEST PASSED, accelerator_test_cuda_not_configured()!!\n");
    } else {
        printf("\n\n\nTEST FAILED, accelerator_test_cuda_not_configured()!!!\n\n\n");
    }
}

static void accelerator_test_get_device(int dev_id) {
    int cuda_dev_id = -1;
    int ret = -1;

    ret = opal_accelerator.get_device(&cuda_dev_id);
    if (OPAL_SUCCESS == ret && cuda_dev_id == dev_id) {
        printf("TEST PASSED, accelerator_test_get_device() succeeded!!!\n");
    } else {
        printf("CUDA DEVICE ID: %d\n", cuda_dev_id);
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_get_device() failed!!!\n\n\n");
    }
}

static void accelerator_test_device_can_access_peer(void) {
    int access = -1;
    int device_count = -1;
    int ret = -1;

    cudaGetDeviceCount(&device_count);

    if (device_count == 1) {
        ret = opal_accelerator.device_can_access_peer(&access, 0, 1);
        if (OPAL_SUCCESS == ret || access == 1) {
            printf("\n\n\nTEST FAILURE!!!! accelerator_test_device_can_access_peer() failed!!!\n\n\n");
        }
    }
    else if (device_count > 1) {
        ret = opal_accelerator.device_can_access_peer(&access, 1, 2);
        if (OPAL_SUCCESS != ret || access != 1) {
            printf("\n\n\nTEST FAILURE!!!! accelerator_test_device_can_access_peer() failed!!!\n\n\n");
        }
    }
    else {
        printf("TEST FAILED:, no CUDA DEVICE!!!");
    }

    // Attempt to self connect (access to connect to self is always 0)
    ret = opal_accelerator.device_can_access_peer(&access, 0, 0);
    if (OPAL_SUCCESS != ret || access != 0) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_device_can_access_peer() failed, access to connect to self non-zero!!!\n\n\n");
    }

    // crazy invalid INPUT
    ret = opal_accelerator.device_can_access_peer(&access, -1, -2);
    printf("\n\n\n ret: %d \n\n\n", ret);
    if (OPAL_SUCCESS == ret || access != 0) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_device_can_access_peer() failed, access to connect to self non-zero!!!\n\n\n");
    }

    printf("TEST PASSED, accelerator_test_device_can_access_peer() succeeded!!!\n");
}

static void accelerator_test_malloc_checkaddr_free(int dev_id) {
    void* ptr = NULL;
    int ret = -1;
    int cuda_dev_id = -1;
    size_t flags = 0;

    // Test malloc
    ret = opal_accelerator.malloc(&ptr, 16);
    if (OPAL_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_malloc_checkaddr_free(): malloc failed!!!\n\n\n");
    }

    // Test check_addr
    flags = 0;
    ret = opal_accelerator.check_addr(ptr, &cuda_dev_id, &flags);
    if (ret <= 0 || dev_id != cuda_dev_id || flags != 0) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_malloc_checkaddr_free(): check_addr failed!!!\n\n\n");
    }

    // Test free
    ret = opal_accelerator.free(ptr);
    if (OPAL_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_malloc_checkaddr_free(): free 1 failed!!!\n\n\n");
    }
    ret = opal_accelerator.free(NULL);
    if (OPAL_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_malloc_checkaddr_free(): free 2 failed!!!\n\n\n");
    }

    printf("TEST PASSED, accelerator_test_malloc_checkaddr_free() succeeded!!!\n");
}

static void accelerator_test_memcpy(int dev_id) {
    void* memory_on_device = NULL;
    int sending_val = 42, receiving_val = -1;
    int ret = -1;

    // Allocate Memory On GPU
    ret = opal_accelerator.malloc(&memory_on_device, sizeof(int));
    if (OPAL_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_memcpy(): malloc failed!!!\n\n\n");
    }

    // Test memcpy
    ret = opal_accelerator.memcpy(dev_id, memory_on_device, &sending_val, sizeof(int), MCA_ACCELERATOR_TRANSFER_HTOD);
    if (OPAL_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_memcpy(): memcpy1 (host to dev) failed!!!\n\n\n");
    }

    // Copy Result Back
    ret = opal_accelerator.memcpy(dev_id, &receiving_val, memory_on_device, sizeof(int), MCA_ACCELERATOR_TRANSFER_DTOH);
    if (OPAL_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_memcpy(): memcpy2 (dev to host)) failed!!!\n\n\n");
    }

    if (sending_val != receiving_val) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_memcpy(): result: %d != expected result: %d (a to dev) failed!!!\n\n\n", sending_val, receiving_val);
    }

    printf("TEST PASSED, accelerator_test_memcpy() succeeded!!!\n");
}


static void accelerator_test_register_memory(void) {
    int rv = -1;
    int* a = malloc(sizeof(int));
    rv = opal_accelerator.host_register(a, sizeof(int));
    if (rv) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_register_memory(): register failed!!!\n\n\n");
    }

    rv = opal_accelerator.host_unregister(a);
    if (rv) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_register_memory(): unregister failed!!!\n\n\n");
    }

    printf("TEST PASSED, accelerator_test_register_memory() succeeded!!!\n");
}

static void accelerator_test_get_address_range(void) {
    int ret = -1;
    size_t size = 10000 * sizeof(int);
    int* memory_on_device = NULL;
    int* new_ptr = NULL;
    size_t new_size = -1;

    // Allocate Memory On GPU
    ret = opal_accelerator.malloc((void**) &memory_on_device, size);
    if (OPAL_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_get_address_range(): malloc failed!!!\n\n\n");
    }

    ret = opal_accelerator.get_address_range((void**) (memory_on_device + 25), (void**) &new_ptr, &new_size);
    if (OPAL_SUCCESS != ret || size != new_size || memory_on_device != new_ptr) {
        printf("new_size: %zu, size: %zu / memory_on_device: %ls, new_ptr: %ls\n", new_size, sizeof(int) * size, memory_on_device, new_ptr);
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_get_address_range(): get_address_range\n\n\n");
    }

    ret = opal_accelerator.get_address_range((void**) (memory_on_device + 25), (void**) &new_ptr, &new_size);
    if (OPAL_SUCCESS != ret || size != new_size || memory_on_device != new_ptr) {
        printf("new_size: %zu, size: %zu / memory_on_device: %ls, new_ptr: %ls\n", new_size, sizeof(int) * size, memory_on_device, new_ptr);
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_get_address_range(): get_address_range\n\n\n");
    }

    printf("TEST PASSED, accelerator_test_get_address_range() succeeded!!!\n");
}

static void accelerator_test_vector_memcpy(int dev_id) {
    // Setup Matrixies
    int ret = -1;
    int rows = 253;
    int cols = 79;
    int dpitch = cols * sizeof(int);
    int spitch = cols * sizeof(int);
    int height = rows;
    int width = cols * sizeof(int);
    int* src = malloc(height * width);
    int* src_rcv = malloc(height * width);
    void* dst = NULL;

    // Initialize src matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            src[i*j + j] = i * j;
            src_rcv[i*j + j] = -1;
        }
    }

    // Allocate Memory On GPU
    ret = opal_accelerator.malloc(&dst, height * width);
    if (OPAL_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_memcpy(): malloc failed!!!\n\n\n");
    }

    // Test memcpy
    ret = opal_accelerator.vector_memcpy(dev_id, dst, dpitch, src, spitch,  width, height, MCA_ACCELERATOR_TRANSFER_HTOD);
    if (OPAL_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_cuda_vector_memcpy(): memcpy1 (host to dev) failed!!!\n\n\n");
    }

    // Copy Result Back
    ret = opal_accelerator.vector_memcpy(dev_id, src_rcv, spitch, dst, dpitch, width, height, MCA_ACCELERATOR_TRANSFER_DTOH);
    if (OPAL_SUCCESS != ret) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_cuda_vector_memcpy(): memcpy2 (dev to host)) failed!!!\n\n\n");
    }

    // Check the matrix for success
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (src[i*j + j] != src_rcv[i*j + j]) {
                printf("\n\n\nTEST FAILURE!!!! accelerator_cuda_vector_memcpy(): memcpy2 (dev to host)) failed!!!\n\n\n");
                break;
            }
        }
    }


    printf("TEST PASSED, accelerator_cuda_vector_memcpy() succeeded!!!\n");
}

static void accelerator_test_ipc_devices(void) {
    // spawn additional processes
    int pid = fork();
    const char* filename = "cuda_handle.txt";
    int rv = -1;
    int dev_id = (pid != 0) ? 0 : 1;
    setup_cuda(dev_id);

    // Setup Initial CPU Memory
    int size = 1000;
    int a[size];
    for (int i = 0; i < size; i++) {
        a[i] = (dev_id != 0) ? size - i : -1;
    }

    // Test IPC
    if (!opal_accelerator.is_ipc_enabled()) {
        printf("\n\n\nTEST FAILURE!!!! accelerator_test_ipc_devices(): IPC not enabled!!!\n\n\n");
    }

    if (pid != 0) {
        int access = -1;

        rv = opal_accelerator.device_can_access_peer(&access, 0, 1);
           if (0 != rv || access != 1) {
            printf("\n\n\nTEST FAILURE!!!! accelerator_test_ipc_devices() failed, devices cannot access each other for IPC!!!\n\n\n");
        }

        void* memory_on_device = NULL;
        rv = opal_accelerator.malloc(&memory_on_device, size * sizeof(int));
        if (0 != rv || memory_on_device == NULL) {
            printf("\n\n\nTEST FAILURE!!!! accelerator_test_ipc_devices(): malloc failed!!!\n\n\n");
        }

        // TODO FIX MEMCPY
        rv = opal_accelerator.memcpy(dev_id, memory_on_device, a, size * sizeof(int), MCA_ACCELERATOR_TRANSFER_HTOD);
        if (0 != rv) {
            printf("\n\n\nTEST FAILURE!!!! accelerator_test_memcpy(): memcpy1 (host to dev) failed!!!\n\n\n");
        }

        CUipcMemHandle handle;
        rv = opal_accelerator.get_handle(memory_on_device, (void**) &handle);
        if (0 != rv) {
            printf("\n\n\nTEST FAILURE!!!! accelerator_test_ipc_devices(): get_handle should fail with NULLs\n\n\n");
        }

        // Write contents to file
        FILE *fp = fopen(filename, "w");

        fwrite((void*) handle.reserved, 1, 64, fp);
        fclose(fp);

        // Sleep for 5 seconds to allow other process to catch up, and write results into our file
        sleep(5);

        // Get Results
        int b[size];
        for (int i = 0; i < b[size]; i++) {
            b[i] = -100;
        }

        fp = fopen(filename, "r");
        if (size != fread((void*) b, sizeof(int), size, fp)) {
            printf("\n\n\nTEST FAILURE!!!! Process %d failed to final result from file\n\n\n", dev_id);
        }
        fclose(fp);

        for (int i = 0; i < size; i++) {
            if (a[i] != b[i]) {
                printf("\n\n\nTEST FAILURE!!!! a[i]: %d != b[i]: %d \n\n\n", a[i], b[i]);
            }
        }
    }
    else {
        // Start with a sleep
        sleep(2);

        // Get the IPC Handle from the file
        FILE *fp = fopen(filename, "r");
        CUipcMemHandle handle;
        if (64 != fread((void*) handle.reserved, 1, 64, fp)) {
            printf("\n\n\nTEST FAILURE!!!! Process %d failed to read handle from file\n\n\n", dev_id);
        }
        fclose(fp);

        void* dev_ptr = NULL;
        rv = opal_accelerator.open_handle((void**) &handle, 0, &dev_ptr);
        if (0 != rv || dev_ptr == NULL) {
            printf("\n\n\nTEST FAILURE!!!! accelerator_test_ipc_devices(): open_handle should not fail\n\n\n");
        }

        rv = opal_accelerator.memcpy(dev_id, a, dev_ptr, size * sizeof(int), MCA_ACCELERATOR_TRANSFER_DTOH);
        if (0 != rv) {
            printf("\n\n\nTEST FAILURE!!!! accelerator_test_memcpy(): memcpy (host to dev) failed!!!\n\n\n");
        }

        rv = opal_accelerator.close_handle(dev_ptr);
        if (0 != rv) {
            printf("\n\n\nTEST FAILURE!!!! accelerator_test_memcpy(): close_handle failed!!!\n\n\n");
        }

        // Write Results back to main process
        fp = fopen(filename, "w");
        if (size != fwrite((void*) a, sizeof(int), 1000, fp)) {
            printf("\n\n\nTEST FAILURE!!!! accelerator_test_ipc_devices(): Failed to copy results to main process\n\n\n");
        }
        fclose(fp);
        cleanup_cuda(dev_id);
        exit(0);
    }

    cleanup_cuda(dev_id);
    printf("TEST PASSED, accelerator_test_ipc_devices() succeeded!!!\n");
}


/* ASYNC Tests */

#define ASSERT(cond) if( !(cond) ) {printf( "ASSERTION FAILURE! line: %d, file(%s)\n", __LINE__, __FILE__ ); fflush(stdout);}


static void accelerator_unit_style_tests(int dev_id)
{
    int rv = -1, size = sizeof(int), dest, src;
    opal_accelerator_stream_t *stream = NULL;
    opal_accelerator_event_t event = {};

    // Successfully Create a stream and event
    ASSERT(0 == opal_accelerator.create_stream(dev_id, &stream));
    ASSERT(0 == opal_accelerator.create_event(&event));

    // FAILURE CASES, Bad Input
    ASSERT(0 != opal_accelerator.create_stream(-1, stream));
    ASSERT(0 != opal_accelerator.create_stream(dev_id, NULL));
    ASSERT(0 == opal_accelerator.destroy_stream(NULL));
    ASSERT(0 != opal_accelerator.synchronize_stream(NULL));
    ASSERT(0 != opal_accelerator.create_event(NULL));
    ASSERT(0 == opal_accelerator.destroy_event(NULL));
    ASSERT(0 != opal_accelerator.record_event(NULL, NULL));
    ASSERT(0 != opal_accelerator.record_event(NULL, stream));
    ASSERT(0 != opal_accelerator.record_event(&event, NULL));
    ASSERT(0 != opal_accelerator.query_event(NULL));
    ASSERT(0 != opal_accelerator.synchronize_event(NULL));
    ASSERT(0 != opal_accelerator.memcpy_async(-1, NULL, NULL, -1, NULL, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.memcpy_async(-1, dest, src, size, stream, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.memcpy_async(dev_id, NULL, src, size, stream, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.memcpy_async(dev_id, dest, NULL, size, stream, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.memcpy_async(dev_id, dest, src, -1, stream, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.memcpy_async(dev_id, dest, src, size, NULL, MCA_ACCELERATOR_TRANSFER_UNSPEC));

    /* NON ASYNC API */
    ASSERT(0 != opal_accelerator.memcpy(-1, (void*)5, (void*) 5, 5, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.memcpy(0, NULL, (void*) 5, 5, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.memcpy(0, (void*)5, NULL, 5, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.vector_memcpy(-1, (void*) 5, 6, (void*) 5, 6, 7, 7, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.vector_memcpy(0, NULL, 6, (void*) 5, 6, 7, 7, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.vector_memcpy(0, (void*) 5, 6, NULL, 6, 7, 7, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.vector_memcpy(0, (void*) 5, 6, (void*) 5, 6, 0, 7, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.vector_memcpy(0, (void*) 5, 6, (void*) 5, 6, 7, 0, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.memmove(-1, (void*)5, (void*) 5, 5, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.memmove(0, NULL, (void*) 5, 5, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.memmove(0, (void*)5, NULL, 5, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    ASSERT(0 != opal_accelerator.malloc(NULL, 5));
    ASSERT(0 != opal_accelerator.malloc((void**)5,  0));
    ASSERT(0 == opal_accelerator.free(NULL));
    ASSERT(0 != opal_accelerator.get_address_range(NULL, NULL, NULL));
    ASSERT(0 != opal_accelerator.get_address_range((void*) 5, (void**) 5, NULL));
    ASSERT(0 != opal_accelerator.get_address_range((void*) 5, NULL, (size_t*) 5));
    ASSERT(0 != opal_accelerator.get_address_range(NULL, (void**) 5, (size_t*) 5));
    ASSERT(0 != opal_accelerator.get_handle(NULL, (void**) 5));
    ASSERT(0 != opal_accelerator.get_handle((void*) 5, NULL));
    ASSERT(0 != opal_accelerator.open_handle(NULL, 0, (void**) 6));
    ASSERT(0 != opal_accelerator.open_handle((void*) 6, 0, NULL));
    ASSERT(0 != opal_accelerator.open_handle((void*) 6, -1, (void**) 7));
    ASSERT(0 == opal_accelerator.close_handle(NULL));
    ASSERT(0 != opal_accelerator.host_register(NULL, 5));
    ASSERT(0 == opal_accelerator.host_unregister(NULL));
    ASSERT(0 != opal_accelerator.get_device(NULL));
    ASSERT(0 != opal_accelerator.device_can_access_peer(NULL, 1, 2));

    // Successfully Destroy a stream and event
    ASSERT(0 == opal_accelerator.destroy_stream(stream));
    ASSERT(0 == opal_accelerator.destroy_event(&event));

    printf("TEST PASSED, accelerator_test_async_unit_style_tests() succeeded!!!\n");
}


static void accelerator_test_empty_synchronizes(int dev_id)
{
    int rv = -1;
    opal_accelerator_stream_t *stream = NULL;
    opal_accelerator_event_t event1 = {}, event2 = {}, event3 = {};

    // Successfully Create a stream and events
    ASSERT(0 == opal_accelerator.create_stream(dev_id, &stream));
    ASSERT(0 == opal_accelerator.create_event(&event1));
    ASSERT(0 == opal_accelerator.create_event(&event2));
    ASSERT(0 == opal_accelerator.create_event(&event3));

    // Query an Event that isn't on a Stream, always complete
    ASSERT(0 == opal_accelerator.query_event(&event3));

    // Empty Syncs
    ASSERT(0 == opal_accelerator.synchronize_stream(stream));     // Sync Empty Stream, shouldn't block (no events on stream)
    ASSERT(0 == opal_accelerator.record_event(&event1, stream));  // Record Event 1
    ASSERT(0 == opal_accelerator.synchronize_stream(stream));     // Sync Stream, Event 1 should be complete
    ASSERT(0 == opal_accelerator.record_event(&event2, stream));  // record event 2,
    ASSERT(0 == opal_accelerator.synchronize_event(&event1));     // sync event 1, already complete
    ASSERT(0 == opal_accelerator.synchronize_event(&event2));     // sync event 2, should be complete
    ASSERT(0 == opal_accelerator.synchronize_event(&event3));     // Event 3 was never added to stream, always complete

    // Re-Add Events to stream
    ASSERT(0 == opal_accelerator.record_event(&event1, stream));
    ASSERT(0 == opal_accelerator.record_event(&event2, stream));
    ASSERT(0 == opal_accelerator.record_event(&event3, stream));

    // Sync Stream, should be fine
    ASSERT(0 == opal_accelerator.synchronize_stream(stream));

    // Re-Add Events to stream
    ASSERT(0 == opal_accelerator.record_event(&event1, stream));
    ASSERT(0 == opal_accelerator.record_event(&event2, stream));
    ASSERT(0 == opal_accelerator.record_event(&event3, stream));

    // Sync Event 3, should block forever
    ASSERT(0 == opal_accelerator.synchronize_event(&event3));     // Event 3 was never added to stream, always complete

    // Successfully Destroy a stream and events
    ASSERT(0 == opal_accelerator.destroy_stream(stream));
    ASSERT(0 == opal_accelerator.destroy_event(&event1));
    ASSERT(0 == opal_accelerator.destroy_event(&event2));
    ASSERT(0 == opal_accelerator.destroy_event(&event3));

    printf("TEST PASSED, accelerator_test_stream() succeeded!!!\n");
}

static void accelerator_test_memcpy_async(int dev_id) {
    int rv = -1;
    int size =  sizeof(int) * 1024 * 1024 * 500;
    int *memory_on_device = NULL, *sending = NULL, *receiving = NULL;
    opal_accelerator_stream_t *stream = NULL, *stream1 = NULL;
    opal_accelerator_event_t event = {};

    // Allocate Host Memory
    sending = (int*) malloc(size);
    receiving = (int*) malloc(size);

    // Set Host Buffers to initial values
    for (int i = 0; i < size / sizeof(int); i++) {
        sending[i] = i;
        receiving[i] = -1;
    }

    // Allocate Memory On GPU
    ASSERT(0 == opal_accelerator.malloc(&memory_on_device, size));

    // Create ASYNC Objects
    ASSERT(0 == opal_accelerator.create_stream(dev_id, &stream));
    ASSERT(0 == opal_accelerator.create_stream(dev_id, &stream1));
    ASSERT(0 == opal_accelerator.create_event(&event));

    // Perform Memcpy from host to device
    ASSERT(0 == opal_accelerator.memcpy_async(dev_id, memory_on_device, sending, size, stream, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    // Add Event to Stream
    ASSERT(0 == opal_accelerator.record_event(&event, stream));
    // sync event 1
    ASSERT(0 == opal_accelerator.synchronize_event(&event));

    // Perform Memcpy from device to host
    ASSERT(0 == opal_accelerator.memcpy_async(dev_id, receiving, memory_on_device, size, stream1, MCA_ACCELERATOR_TRANSFER_UNSPEC));
    // Add Event to Stream
    ASSERT(0 == opal_accelerator.record_event(&event, stream));
    // sync event 1
    ASSERT(0 == opal_accelerator.synchronize_stream(stream));

    // Test Result
    for (int i = 0; i < size / sizeof(int); i++) {
        if(sending[i] != receiving[i]) {
            printf("\n\ni: %d, sending[i]: %d, receiving[i]: %d \n\n", i, sending[i], receiving[i]);
            ASSERT(0);
            break;
        }
    }

    ASSERT(0 == opal_accelerator.free(memory_on_device));
    ASSERT(0 == opal_accelerator.destroy_stream(stream));
    ASSERT(0 == opal_accelerator.destroy_stream(stream1));
    ASSERT(0 == opal_accelerator.destroy_event(&event));

    printf("TEST PASSED, accelerator_test_memcpy_async() succeeded!!!\n");
}

static int
mca_pml_cm_component_open(void)
{
    int ret;

    // TESTING HAPPENS HERE ... but only on rank 0
    if (omb_get_local_rank() == 0) {
        int dev_id = 0;
        printf("\n\n Starting CUDA TESTS \n\n");
        setup_accelerator();

        // Run Tests that don't require CUDA to be setup
        // accelerator_test_ipc_devices();
        //accelerator_test_cuda_not_configured();

        // setup_cuda(dev_id);

        // Only accelerator_test_ipc_devices or accelerator_test_memcpy work, to test this
        // comment out IPC test and vice versa
        accelerator_test_memcpy(dev_id);

        accelerator_test_get_device(dev_id);
        // accelerator_test_device_can_access_peer();
        accelerator_test_malloc_checkaddr_free(dev_id);
        accelerator_test_register_memory();
        accelerator_test_get_address_range();
        accelerator_test_vector_memcpy(dev_id);

        // ASYNC Tests:
        accelerator_unit_style_tests(dev_id);
        accelerator_test_empty_synchronizes(dev_id);

        // for (int i = 0; i < 100; i++) {
            accelerator_test_memcpy_async(dev_id);
        // }


        cleanup_accelerator();
        printf("\n\n Finished CUDA TESTS \n\n");
    }



    ret = mca_base_framework_open(&ompi_mtl_base_framework, 0);
    if (OMPI_SUCCESS == ret) {
      /* If no MTL components initialized CM component can be unloaded */
      if (0 == opal_list_get_size(&ompi_mtl_base_framework.framework_components)) {
    ret = OPAL_ERR_NOT_AVAILABLE;
      }
    }

    return ret;
}


static int
mca_pml_cm_component_close(void)
{
    return mca_base_framework_close(&ompi_mtl_base_framework);
}


static mca_pml_base_module_t*
mca_pml_cm_component_init(int* priority,
                          bool enable_progress_threads,
                          bool enable_mpi_threads)
{
    int ret;

    *priority = -1;

    opal_output_verbose( 10, 0,
                         "in cm pml priority is %d\n", *priority);
    /* find a useable MTL */
    ret = ompi_mtl_base_select(enable_progress_threads, enable_mpi_threads, priority);
    if (OMPI_SUCCESS != ret) {
        return NULL;
    }

    if (ompi_mtl->mtl_flags & MCA_MTL_BASE_FLAG_REQUIRE_WORLD) {
        ompi_pml_cm.super.pml_flags |= MCA_PML_BASE_FLAG_REQUIRE_WORLD;
    }

    if (ompi_mtl->mtl_flags & MCA_MTL_BASE_FLAG_SUPPORTS_EXT_CID) {
        ompi_pml_cm.super.pml_flags |= MCA_PML_BASE_FLAG_SUPPORTS_EXT_CID;
    }

    ompi_pml_cm.super.pml_max_contextid = ompi_mtl->mtl_max_contextid;
    ompi_pml_cm.super.pml_max_tag = ompi_mtl->mtl_max_tag;

    return &ompi_pml_cm.super;
}


static int
mca_pml_cm_component_fini(void)
{
    if (NULL != ompi_mtl) {
        return OMPI_MTL_CALL(finalize(ompi_mtl));
    }

    return OMPI_SUCCESS;
}

