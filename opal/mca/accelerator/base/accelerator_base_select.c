/*
 * Copyright (c) 2004-2010 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2007 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2012-2015 Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2013-2020 Intel, Inc.  All rights reserved.
 * Copyright (c) 2015-2020 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2020-2022 Amazon.com, Inc. or its affiliates.  All Rights
 * Copyright (c) 2018-2020 Triad National Security, LLC. All rights
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"
#include "opal/class/opal_list.h"
#include "opal/util/output.h"
#include "opal/util/show_help.h"
#include "opal/util/proc.h"
#include "opal/constants.h"
#include "opal/mca/base/base.h"
#include "opal/mca/mca.h"
#include "opal/mca/accelerator/base/base.h"
#include "opal/mca/accelerator/accelerator.h"
#include <string.h>

typedef struct accelerator_component_t {
  opal_list_item_t super;
  opal_accelerator_base_component_t *accelerator_component;
} accelerator_component_t;

int opal_accelerator_base_select(void)
{
    mca_base_component_list_item_t *cli = NULL;
    opal_list_item_t *item = NULL;
    opal_accelerator_base_component_t *component = NULL;
    opal_accelerator_base_module_t *module = NULL;
    accelerator_component_t *ac = NULL;
    opal_list_t ordered_list;
    bool found_component = false;

    OBJ_CONSTRUCT(&ordered_list, opal_list_t);

    /* Traverse the list of available components and create a new ordered list with
       the NULL Component on the back of the new list */
    OPAL_LIST_FOREACH(cli, &opal_accelerator_base_framework.framework_components, mca_base_component_list_item_t) {
        ac = (accelerator_component_t*) malloc(sizeof(accelerator_component_t));
        if (NULL == ac) {
            return OPAL_ERR_OUT_OF_RESOURCE;
        }

        OBJ_CONSTRUCT(ac, opal_list_item_t);
        ac->accelerator_component = (opal_accelerator_base_component_t *) cli->cli_component;;

        if (0 == strcmp(cli->cli_component->mca_component_name, "null")) {
            opal_list_append(&ordered_list, (opal_list_item_t*) ac);
        }
        else {
            opal_list_prepend(&ordered_list, (opal_list_item_t*) ac);
        }
    }

    /* Traverse the ordered list of available components and try and initialize every component
     * in the list. If more than 1 (non NULL) components initialize successfully, abort MPI run.
     * Return the NULL component if no other components initialize successfully */
    for (item = opal_list_remove_first(&ordered_list); NULL != item; item = opal_list_remove_first(&ordered_list)) {
        component = ((accelerator_component_t *) item)->accelerator_component;
        if (NULL == component->accelerator_init) {
            opal_output_verbose(10, opal_accelerator_base_framework.framework_output,
                                 "select: no init function; ignoring component %s",
                                 component->base_version.mca_component_name);
            continue;
        }

        opal_output_verbose(10, opal_accelerator_base_framework.framework_output,
                            "select: initializing %s component %s",
                            component->base_version.mca_type_name,
                            component->base_version.mca_component_name);

        module = component->accelerator_init();

        if (NULL != module) {
            if (found_component) {
                if (0 != ordered_list.opal_list_length) {
                    /* Handle the error case where two non-null components initialize successfully */
                    opal_show_help("help-accelerator-base.txt", "Multiple Accelerators Found", true,
                                   accelerator_base_selected_component.base_version.mca_component_name,
                                   component->base_version.mca_component_name);

                    /* Don't leak the two initalized components */
                    accelerator_base_selected_component.accelerator_finalize(&opal_accelerator);
                    component->accelerator_finalize(module);

                    /* Throw a fatal error, can not go on without knowing which Accelerator to use
                     * Leak some memory but doesn't matter b/c OMPI is aborting */
                    return OPAL_ERR_FATAL;
                }

                /* Prefer found component over NULL component*/
                OBJ_DESTRUCT(item);
                free(item);
                continue;
            }

            found_component = true;
            accelerator_base_selected_component = *component;
            opal_accelerator = *module;
        }

        OBJ_DESTRUCT(item);
        free(item);
    }

    OBJ_DESTRUCT(&ordered_list);

    /* This base function closes, unloads, and removes from the
     * available list all unselected components.  The available list will
     * contain only the selected component. */

    mca_base_components_close(opal_accelerator_base_framework.framework_output,
                              &opal_accelerator_base_framework.framework_components,
                              (mca_base_component_t *) &accelerator_base_selected_component);

    opal_output_verbose(10, opal_accelerator_base_framework.framework_output, "selected %s\n",
                        accelerator_base_selected_component.base_version.mca_component_name);

    return OPAL_SUCCESS;
}
